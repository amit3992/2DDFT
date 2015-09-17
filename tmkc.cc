Complex val = 0;
int N = w;
double theta = 0.0;
for (int n=0; n < N; n++)
{
  Complex D = 0;
  for(int k=0; k < N; k++)
  {
    theta = ((2*M_PI*n*k)/N);
    val = Complex(cos(theta), -1*sin(theta));
    D = D + (h[k]*val);
  }
    H[n] = D;
}


// ================ VIGNESH ==========================//

for(int i = 0; i < w; ++i) {
  Complex temp;
  Complex W_temp;

  for (int j = 0; j < w; j++) {
    W_temp = Complex(cos((i*j*2*M_PI)/w),-sin((i*j*2*M_PI)/w));
    temp = temp + (W_temp * h[j]);
  }
  H[i] = temp;
}
