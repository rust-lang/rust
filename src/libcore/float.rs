#[doc = "Operations and constants for `float`"];

// Even though this module exports everything defined in it,
// because it contains re-exports, we also have to explicitly
// export locally defined things. That's a bit annoying.
export to_str_common, to_str_exact, to_str, from_str;
export add, sub, mul, div, rem, lt, le, gt, eq, eq, ne;
export is_positive, is_negative, is_nonpositive, is_nonnegative;
export is_zero, is_infinite, is_finite;
export NaN, is_NaN, infinity, neg_infinity;
export consts;
export logarithm;
export acos, asin, atan, atan2, cbrt, ceil, copysign, cos, cosh, floor;
export erf, erfc, exp, expm1, exp2, abs, abs_sub;
export mul_add, fmax, fmin, nextafter, frexp, hypot, ldexp;
export lgamma, ln, log_radix, ln1p, log10, log2, ilog_radix;
export modf, pow, round, sin, sinh, sqrt, tan, tanh, tgamma, trunc;
export signbit;
export pow_with_uint;
export extensions;

// export when m_float == c_double

export j0, j1, jn, y0, y1, yn;

// PORT this must match in width according to architecture

import m_float = f64;
import f64::*;
import num::num;

const NaN: float = 0.0/0.0;

const infinity: float = 1.0/0.0;

const neg_infinity: float = -1.0/0.0;

/* Module: consts */
mod consts {

    // FIXME (requires Issue #1433 to fix): replace with mathematical
    // constants from cmath.
    #[doc = "Archimedes' constant"]
    const pi: float = 3.14159265358979323846264338327950288;

    #[doc = "pi/2.0"]
    const frac_pi_2: float = 1.57079632679489661923132169163975144;

    #[doc = "pi/4.0"]
    const frac_pi_4: float = 0.785398163397448309615660845819875721;

    #[doc = "1.0/pi"]
    const frac_1_pi: float = 0.318309886183790671537767526745028724;

    #[doc = "2.0/pi"]
    const frac_2_pi: float = 0.636619772367581343075535053490057448;

    #[doc = "2.0/sqrt(pi)"]
    const frac_2_sqrtpi: float = 1.12837916709551257389615890312154517;

    #[doc = "sqrt(2.0)"]
    const sqrt2: float = 1.41421356237309504880168872420969808;

    #[doc = "1.0/sqrt(2.0)"]
    const frac_1_sqrt2: float = 0.707106781186547524400844362104849039;

    #[doc = "Euler's number"]
    const e: float = 2.71828182845904523536028747135266250;

    #[doc = "log2(e)"]
    const log2_e: float = 1.44269504088896340735992468100189214;

    #[doc = "log10(e)"]
    const log10_e: float = 0.434294481903251827651128918916605082;

    #[doc = "ln(2.0)"]
    const ln_2: float = 0.693147180559945309417232121458176568;

    #[doc = "ln(10.0)"]
    const ln_10: float = 2.30258509299404568401799145468436421;
}

/**
 * Section: String Conversions
 */

#[doc = "
Converts a float to a string

# Arguments

* num - The float value
* digits - The number of significant digits
* exact - Whether to enforce the exact number of significant digits
"]
fn to_str_common(num: float, digits: uint, exact: bool) -> str {
    if is_NaN(num) { ret "NaN"; }
    if num == infinity { ret "inf"; }
    if num == neg_infinity { ret "-inf"; }

    let mut (num, sign) = if num < 0.0 { (-num, "-") } else { (num, "") };

    // truncated integer
    let trunc = num as uint;

    // decimal remainder
    let mut frac = num - (trunc as float);

    // stack of digits
    let mut fractionalParts = [];

    // FIXME: (#2608)
    // This used to return right away without rounding, as "[-]num",
    // but given epsilon like in f64.rs, I don't see how the comparison
    // to epsilon did much when only used there.
    //    if (frac < epsilon && !exact) || digits == 0u { ret accum; }
    //
    // With something better, possibly weird results like this can be avoided:
    //     assert "3.14158999999999988262" == my_to_str_exact(3.14159, 20u);

    let mut ii = digits;
    let mut epsilon_prime = 1.0 / pow_with_uint(10u, ii);

    // while we still need digits
    // build stack of digits
    while ii > 0u && (frac >= epsilon_prime || exact) {
        // store the next digit
        frac *= 10.0;
        let digit = frac as uint;
        vec::push(fractionalParts, digit);

        // calculate the next frac
        frac -= digit as float;
        epsilon_prime *= 10.0;
        ii -= 1u;
    }

    let mut acc;
    let mut racc = "";
    let mut carry = if frac * 10.0 as uint >= 5u { 1u } else { 0u };

    // turn digits into string
    // using stack of digits
    while vec::len(fractionalParts) > 0u {
        let mut adjusted_digit = carry + vec::pop(fractionalParts);

        if adjusted_digit == 10u {
            carry = 1u;
            adjusted_digit %= 10u
        } else {
            carry = 0u
        };

        racc = uint::str(adjusted_digit) + racc;
    }

    // pad decimals with trailing zeroes
    while str::len(racc) < digits && exact {
        racc += "0"
    }

    // combine ints and decimals
    let mut ones = uint::str(trunc + carry);
    if racc == "" {
        acc = sign + ones;
    } else {
        acc = sign + ones + "." + racc;
    }

    ret acc;
}

#[doc = "
Converts a float to a string with exactly the number of
provided significant digits

# Arguments

* num - The float value
* digits - The number of significant digits
"]
fn to_str_exact(num: float, digits: uint) -> str {
    to_str_common(num, digits, true)
}

#[test]
fn test_to_str_exact_do_decimal() {
    let s = to_str_exact(5.0, 4u);
    assert s == "5.0000";
}


#[doc = "
Converts a float to a string with a maximum number of
significant digits

# Arguments

* num - The float value
* digits - The number of significant digits
"]
fn to_str(num: float, digits: uint) -> str {
    to_str_common(num, digits, false)
}

#[doc = "
Convert a string to a float

This function accepts strings such as

* '3.14'
* '+3.14', equivalent to '3.14'
* '-3.14'
* '2.5E10', or equivalently, '2.5e10'
* '2.5E-10'
* '', or, equivalently, '.' (understood as 0)
* '5.'
* '.5', or, equivalently,  '0.5'
* 'inf', '-inf', 'NaN'

Leading and trailing whitespace are ignored.

# Arguments

* num - A string

# Return value

`none` if the string did not represent a valid number.  Otherwise, `some(n)`
where `n` is the floating-point number represented by `[num]`.
"]
fn from_str(num: str) -> option<float> {
   if num == "inf" {
       ret some(infinity as float);
   } else if num == "-inf" {
       ret some(neg_infinity as float);
   } else if num == "NaN" {
       ret some(NaN as float);
   }

   let mut pos = 0u;               //Current byte position in the string.
                                   //Used to walk the string in O(n).
   let len = str::len(num);        //Length of the string, in bytes.

   if len == 0u { ret none; }
   let mut total = 0f;             //Accumulated result
   let mut c     = 'z';            //Latest char.

   //The string must start with one of the following characters.
   alt str::char_at(num, 0u) {
      '-' | '+' | '0' to '9' | '.' {}
      _ { ret none; }
   }

   //Determine if first char is '-'/'+'. Set [pos] and [neg] accordingly.
   let mut neg = false;               //Sign of the result
   alt str::char_at(num, 0u) {
      '-' {
          neg = true;
          pos = 1u;
      }
      '+' {
          pos = 1u;
      }
      _ {}
   }

   //Examine the following chars until '.', 'e', 'E'
   while(pos < len) {
       let char_range = str::char_range_at(num, pos);
       c   = char_range.ch;
       pos = char_range.next;
       alt c {
         '0' to '9' {
           total = total * 10f;
           total += ((c as int) - ('0' as int)) as float;
         }
         '.' | 'e' | 'E' {
           break;
         }
         _ {
           ret none;
         }
       }
   }

   if c == '.' {//Examine decimal part
      let mut decimal = 1f;
      while(pos < len) {
         let char_range = str::char_range_at(num, pos);
         c = char_range.ch;
         pos = char_range.next;
         alt c {
            '0' | '1' | '2' | '3' | '4' | '5' | '6'| '7' | '8' | '9'  {
                 decimal /= 10f;
                 total += (((c as int) - ('0' as int)) as float)*decimal;
             }
             'e' | 'E' {
                 break;
             }
             _ {
                 ret none;
             }
         }
      }
   }

   if (c == 'e') | (c == 'E') {//Examine exponent
      let mut exponent = 0u;
      let mut neg_exponent = false;
      if(pos < len) {
          let char_range = str::char_range_at(num, pos);
          c   = char_range.ch;
          alt c  {
             '+' {
                pos = char_range.next;
             }
             '-' {
                pos = char_range.next;
                neg_exponent = true;
             }
             _ {}
          }
          while(pos < len) {
             let char_range = str::char_range_at(num, pos);
             c = char_range.ch;
             alt c {
                 '0' | '1' | '2' | '3' | '4' | '5' | '6'| '7' | '8' | '9' {
                     exponent *= 10u;
                     exponent += ((c as uint) - ('0' as uint));
                 }
                 _ {
                     break;
                 }
             }
             pos = char_range.next;
          }
          let multiplier = pow_with_uint(10u, exponent);
              //Note: not [int::pow], otherwise, we'll quickly
              //end up with a nice overflow
          if neg_exponent {
             total = total / multiplier;
          } else {
             total = total * multiplier;
          }
      } else {
         ret none;
      }
   }

   if(pos < len) {
     ret none;
   } else {
     if(neg) {
        total *= -1f;
     }
     ret some(total);
   }
}

/**
 * Section: Arithmetics
 */

#[doc = "
Compute the exponentiation of an integer by another integer as a float

# Arguments

* x - The base
* pow - The exponent

# Return value

`NaN` if both `x` and `pow` are `0u`, otherwise `x^pow`
"]
fn pow_with_uint(base: uint, pow: uint) -> float {
    if base == 0u {
        if pow == 0u {
            ret NaN as float;
        }
        ret 0.;
    }
    let mut my_pow     = pow;
    let mut total      = 1f;
    let mut multiplier = base as float;
    while (my_pow > 0u) {
        if my_pow % 2u == 1u {
            total = total * multiplier;
        }
        my_pow     /= 2u;
        multiplier *= multiplier;
    }
    ret total;
}

fn is_positive(x: float) -> bool { f64::is_positive(x as f64) }
fn is_negative(x: float) -> bool { f64::is_negative(x as f64) }
fn is_nonpositive(x: float) -> bool { f64::is_nonpositive(x as f64) }
fn is_nonnegative(x: float) -> bool { f64::is_nonnegative(x as f64) }
fn is_zero(x: float) -> bool { f64::is_zero(x as f64) }
fn is_infinite(x: float) -> bool { f64::is_infinite(x as f64) }
fn is_finite(x: float) -> bool { f64::is_finite(x as f64) }
fn is_NaN(x: float) -> bool { f64::is_NaN(x as f64) }

fn abs(x: float) -> float { f64::abs(x as f64) as float }
fn sqrt(x: float) -> float { f64::sqrt(x as f64) as float }
fn atan(x: float) -> float { f64::atan(x as f64) as float }
fn sin(x: float) -> float { f64::sin(x as f64) as float }
fn cos(x: float) -> float { f64::cos(x as f64) as float }
fn tan(x: float) -> float { f64::tan(x as f64) as float }

mod extensions {
    impl num of num for float {
        fn add(&&other: float)    -> float { ret self + other; }
        fn sub(&&other: float)    -> float { ret self - other; }
        fn mul(&&other: float)    -> float { ret self * other; }
        fn div(&&other: float)    -> float { ret self / other; }
        fn modulo(&&other: float) -> float { ret self % other; }
        fn neg()                  -> float { ret -self;        }

        fn to_int()         -> int   { ret self as int; }
        fn from_int(n: int) -> float { ret n as float;  }
    }
}

#[test]
fn test_from_str() {
   assert from_str("3") == some(3.);
   assert from_str("3") == some(3.);
   assert from_str("3.14") == some(3.14);
   assert from_str("+3.14") == some(3.14);
   assert from_str("-3.14") == some(-3.14);
   assert from_str("2.5E10") == some(25000000000.);
   assert from_str("2.5e10") == some(25000000000.);
   assert from_str("25000000000.E-10") == some(2.5);
   assert from_str(".") == some(0.);
   assert from_str(".e1") == some(0.);
   assert from_str(".e-1") == some(0.);
   assert from_str("5.") == some(5.);
   assert from_str(".5") == some(0.5);
   assert from_str("0.5") == some(0.5);
   assert from_str("0.5") == some(0.5);
   assert from_str("0.5") == some(0.5);
   assert from_str("-.5") == some(-0.5);
   assert from_str("-.5") == some(-0.5);
   assert from_str("-5") == some(-5.);
   assert from_str("-0") == some(-0.);
   assert from_str("0") == some(0.);
   assert from_str("inf") == some(infinity);
   assert from_str("-inf") == some(neg_infinity);
   // note: NaN != NaN, hence this slightly complex test
   alt from_str("NaN") {
       some(f) { assert is_NaN(f); }
       none { fail; }
   }

   assert from_str("") == none;
   assert from_str("x") == none;
   assert from_str(" ") == none;
   assert from_str("   ") == none;
   assert from_str("e") == none;
   assert from_str("E") == none;
   assert from_str("E1") == none;
   assert from_str("1e1e1") == none;
   assert from_str("1e1.1") == none;
   assert from_str("1e1-1") == none;
}

#[test]
fn test_positive() {
  assert(is_positive(infinity));
  assert(is_positive(1.));
  assert(is_positive(0.));
  assert(!is_positive(-1.));
  assert(!is_positive(neg_infinity));
  assert(!is_positive(1./neg_infinity));
  assert(!is_positive(NaN));
}

#[test]
fn test_negative() {
  assert(!is_negative(infinity));
  assert(!is_negative(1.));
  assert(!is_negative(0.));
  assert(is_negative(-1.));
  assert(is_negative(neg_infinity));
  assert(is_negative(1./neg_infinity));
  assert(!is_negative(NaN));
}

#[test]
fn test_nonpositive() {
  assert(!is_nonpositive(infinity));
  assert(!is_nonpositive(1.));
  assert(!is_nonpositive(0.));
  assert(is_nonpositive(-1.));
  assert(is_nonpositive(neg_infinity));
  assert(is_nonpositive(1./neg_infinity));
  assert(!is_nonpositive(NaN));
}

#[test]
fn test_nonnegative() {
  assert(is_nonnegative(infinity));
  assert(is_nonnegative(1.));
  assert(is_nonnegative(0.));
  assert(!is_nonnegative(-1.));
  assert(!is_nonnegative(neg_infinity));
  assert(!is_nonnegative(1./neg_infinity));
  assert(!is_nonnegative(NaN));
}

#[test]
fn test_to_str_inf() {
    assert to_str(infinity, 10u) == "inf";
    assert to_str(-infinity, 10u) == "-inf";
}

#[test]
fn test_ifaces() {
    fn test<U:num>(ten: U) {
        assert (ten.to_int() == 10);

        let two = ten.from_int(2);
        assert (two.to_int() == 2);

        assert (ten.add(two) == ten.from_int(12));
        assert (ten.sub(two) == ten.from_int(8));
        assert (ten.mul(two) == ten.from_int(20));
        assert (ten.div(two) == ten.from_int(5));
        assert (ten.modulo(two) == ten.from_int(0));
    }

    test(10.0);
}


//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//





