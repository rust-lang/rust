/* On platforms that don't support weak symbols, define required aliases
 * as wrappers. See comments in `features.h` for more.
 */
#if defined(__APPLE__) || defined(__MINGW32__)

double __lgamma_r(double a, int *b);
float __lgammaf_r(float a, int *b);
long __lgammal_r(long double a, int *b);
double exp10(double a);
float exp10f(float a);
long exp10l(long double a);
double remainder(double a, double b);
float remainderf(float a, float b);

double lgamma_r(double a, int *b) {
	return __lgamma_r(a, b);
}
float lgammaf_r(float a, int *b) {
	return __lgammaf_r(a, b);
}
long double lgammal_r(long double a, int *b) {
	return __lgammal_r(a, b);
}
double pow10(double a) {
	return exp10(a);
}
float pow10f(float a) {
	return exp10f(a);
}
long double pow10l(long double a) {
	return exp10l(a);
}
double drem(double a, double b) {
	return remainder(a, b);
}
float dremf(float a, float b) {
	return remainderf(a, b);
}

#endif
