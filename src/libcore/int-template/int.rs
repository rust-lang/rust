type T = int;
const bits: uint = uint::bits;

/// Produce a uint suitable for use in a hash table
pure fn hash(x: &int) -> uint {
    let u : uint = *x as uint;
    uint::hash(&u)
}

/// Returns `base` raised to the power of `exponent`
fn pow(base: int, exponent: uint) -> int {
    if exponent == 0u { return 1; } //Not mathemtically true if ~[base == 0]
    if base     == 0  { return 0; }
    let mut my_pow  = exponent;
    let mut acc     = 1;
    let mut multiplier = base;
    while(my_pow > 0u) {
      if my_pow % 2u == 1u {
         acc *= multiplier;
      }
      my_pow     /= 2u;
      multiplier *= multiplier;
    }
    return acc;
}

#[test]
fn test_pow() {
    assert (pow(0, 0u) == 1);
    assert (pow(0, 1u) == 0);
    assert (pow(0, 2u) == 0);
    assert (pow(-1, 0u) == 1);
    assert (pow(1, 0u) == 1);
    assert (pow(-3, 2u) == 9);
    assert (pow(-3, 3u) == -27);
    assert (pow(4, 9u) == 262144);
}

#[test]
fn test_overflows() {
   assert (max_value > 0);
   assert (min_value <= 0);
   assert (min_value + max_value + 1 == 0);
}
