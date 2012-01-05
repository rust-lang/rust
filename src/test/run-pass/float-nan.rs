use std;
import float;

fn main() {
  let nan = float::NaN;
  assert(float::is_NaN(nan));

  let inf = float::infinity;
  assert(-inf == float::neg_infinity);

  assert( nan !=  nan);
  assert( nan != -nan);
  assert(-nan != -nan);
  assert(-nan !=  nan);

  assert( nan !=   1.);
  assert( nan !=   0.);
  assert( nan !=  inf);
  assert( nan != -inf);

  assert(  1. !=  nan);
  assert(  0. !=  nan);
  assert( inf !=  nan);
  assert(-inf !=  nan);

  assert(!( nan ==  nan));
  assert(!( nan == -nan));
  assert(!( nan ==   1.));
  assert(!( nan ==   0.));
  assert(!( nan ==  inf));
  assert(!( nan == -inf));
  assert(!(  1. ==  nan));
  assert(!(  0. ==  nan));
  assert(!( inf ==  nan));
  assert(!(-inf ==  nan));
  assert(!(-nan ==  nan));
  assert(!(-nan == -nan));

  assert(!( nan >  nan));
  assert(!( nan > -nan));
  assert(!( nan >   0.));
  assert(!( nan >  inf));
  assert(!( nan > -inf));
  assert(!(  0. >  nan));
  assert(!( inf >  nan));
  assert(!(-inf >  nan));
  assert(!(-nan >  nan));

  assert(!(nan <   0.));
  assert(!(nan <   1.));
  assert(!(nan <  -1.));
  assert(!(nan <  inf));
  assert(!(nan < -inf));
  assert(!(nan <  nan));
  assert(!(nan < -nan));

  assert(!(  0. < nan));
  assert(!(  1. < nan));
  assert(!( -1. < nan));
  assert(!( inf < nan));
  assert(!(-inf < nan));
  assert(!(-nan < nan));

  assert(float::is_NaN(nan + inf));
  assert(float::is_NaN(nan + -inf));
  assert(float::is_NaN(nan + 0.));
  assert(float::is_NaN(nan + 1.));
  assert(float::is_NaN(nan * 1.));
  assert(float::is_NaN(nan / 1.));
  assert(float::is_NaN(nan / 0.));
  assert(float::is_NaN(0. / 0.));
  assert(float::is_NaN(-inf + inf));
  assert(float::is_NaN(inf - inf));

  assert(!float::is_NaN(-1.));
  assert(!float::is_NaN(0.));
  assert(!float::is_NaN(0.1));
  assert(!float::is_NaN(1.));
  assert(!float::is_NaN(inf));
  assert(!float::is_NaN(-inf));
  assert(!float::is_NaN(1./-inf));
}
