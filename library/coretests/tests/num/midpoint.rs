//! Test the following expectations:
//!  - midpoint(a, b) == (a + b) / 2
//!  - midpoint(a, b) == midpoint(b, a)
//!  - midpoint(-a, -b) == -midpoint(a, b)

#[test]
#[cfg(not(miri))]
fn midpoint_obvious_impl_i8() {
    for a in i8::MIN..=i8::MAX {
        for b in i8::MIN..=i8::MAX {
            assert_eq!(i8::midpoint(a, b), ((a as i16 + b as i16) / 2) as i8);
        }
    }
}

#[test]
#[cfg(not(miri))]
fn midpoint_obvious_impl_u8() {
    for a in u8::MIN..=u8::MAX {
        for b in u8::MIN..=u8::MAX {
            assert_eq!(u8::midpoint(a, b), ((a as u16 + b as u16) / 2) as u8);
        }
    }
}

#[test]
#[cfg(not(miri))]
fn midpoint_order_expectation_i8() {
    for a in i8::MIN..=i8::MAX {
        for b in i8::MIN..=i8::MAX {
            assert_eq!(i8::midpoint(a, b), i8::midpoint(b, a));
        }
    }
}

#[test]
#[cfg(not(miri))]
fn midpoint_order_expectation_u8() {
    for a in u8::MIN..=u8::MAX {
        for b in u8::MIN..=u8::MAX {
            assert_eq!(u8::midpoint(a, b), u8::midpoint(b, a));
        }
    }
}

#[test]
#[cfg(not(miri))]
fn midpoint_negative_expectation() {
    for a in 0..=i8::MAX {
        for b in 0..=i8::MAX {
            assert_eq!(i8::midpoint(-a, -b), -i8::midpoint(a, b));
        }
    }
}
