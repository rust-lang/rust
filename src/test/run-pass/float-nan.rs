use std;
import std::float;

fn main() {
  let nan = float::NaN();
  assert(float::isNaN(nan));

  let inf = float::infinity();
  assert(-inf == float::neg_infinity());

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

  assert(float::isNaN(nan + inf));
  assert(float::isNaN(nan + -inf));
  assert(float::isNaN(nan + 0.));
  assert(float::isNaN(nan + 1.));
  assert(float::isNaN(nan * 1.));
  assert(float::isNaN(nan / 1.));
  assert(float::isNaN(nan / 0.));
  assert(float::isNaN(0. / 0.));
  assert(float::isNaN(-inf + inf));
  assert(float::isNaN(inf - inf));

  assert(!float::isNaN(-1.));
  assert(!float::isNaN(0.));
  assert(!float::isNaN(0.1));
  assert(!float::isNaN(1.));
  assert(!float::isNaN(inf));
  assert(!float::isNaN(-inf));
  assert(!float::isNaN(1./-inf));
}
