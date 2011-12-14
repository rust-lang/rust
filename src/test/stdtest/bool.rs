import core::*;

use std;

import bool;

#[test]
fn test_bool_from_str() {
    bool::all_values { |v|
        assert v == bool::from_str(bool::to_str(v))
    }
}

#[test]
fn test_bool_to_str() {
    assert bool::to_str(false) == "false";
    assert bool::to_str(true) == "true";
}

#[test]
fn test_bool_to_bit() {
    bool::all_values { |v|
        assert bool::to_bit(v) == if bool::is_true(v) { 1u8 } else { 0u8 };
    }
}