import core::*;

use std;

import std::tri;
import std::four;

fn eq1(a: four::t, b: four::t) -> bool { four::eq(a , b) }
fn ne1(a: four::t, b: four::t) -> bool { four::ne(a , b) }

fn eq2(a: four::t, b: four::t) -> bool { eq1( a, b ) && eq1( b, a ) }

#[test]
fn test_four_req_eq() {
    four::all_values { |a|
        four::all_values { |b|
            assert if a == b { eq1( a, b ) } else { ne1( a, b ) };
        }
    }
}

#[test]
fn test_four_and_symmetry() {
    four::all_values { |a|
        four::all_values { |b|
            assert eq1( four::and(a ,b), four::and(b, a) );
        }
    }
}

#[test]
fn test_four_xor_symmetry() {
    four::all_values { |a|
        four::all_values { |b|
            assert eq1( four::and(a ,b), four::and(b, a) );
        }
    }
}

#[test]
fn test_four_or_symmetry() {
    four::all_values { |a|
        four::all_values { |b|
            assert eq1( four::or(a ,b), four::or(b, a) );
        }
    }
}

fn to_tup(v: four::t) -> (bool, bool) {
    alt v {
      0u8 { (false, false) }
      1u8 { (false, true) }
      2u8 { (true, false) }
      3u8 { (true, true) }
    }
}

#[test]
fn test_four_not() {
    four::all_values { |a|
        let (x, y) = to_tup(a);
        assert to_tup(four::not(a)) == (y, x);
    };
}


#[test]
fn test_four_and() {
    four::all_values { |a|
        four::all_values { |b|
            let (y1, x1) = to_tup(a);
            let (y2, x2) = to_tup(b);
            let (y3, x3) = to_tup(four::and(a, b));

            assert (x3, y3) == (x1 && x2, y1 || y2);
        }
    };
}

#[test]
fn test_four_or() {
    four::all_values { |a|
        four::all_values { |b|
            let (y1, x1) = to_tup(a);
            let (y2, x2) = to_tup(b);
            let (y3, x3) = to_tup(four::or(a, b));

            assert (x3, y3) == (x1 || x2, y1 && y2);
        }
    };
}

#[test]
fn test_four_implies() {
    four::all_values { |a|
        four::all_values { |b|
            let (_, x1) = to_tup(a);
            let (y2, x2) = to_tup(b);
            let (y3, x3) = to_tup(four::implies(a, b));

            assert (x3, y3) == (!x1 || x2, x1 && y2);
        }
    };
}

#[test]
fn test_four_is_true() {
    assert !four::is_true(four::none);
    assert !four::is_true(four::false);
    assert four::is_true(four::true);
    assert four::is_true(four::both);
}

#[test]
fn test_four_is_false() {
    assert four::is_false(four::none);
    assert four::is_false(four::false);
    assert !four::is_false(four::true);
    assert !four::is_false(four::both);
}

#[test]
fn test_four_from_str() {
    four::all_values { |v|
        assert eq1( v, four::from_str(four::to_str(v)) );
    }
}

#[test]
fn test_four_to_str() {
    assert four::to_str(four::none) == "none";
    assert four::to_str(four::false) == "false";
    assert four::to_str(four::true) == "true" ;
    assert four::to_str(four::both) == "both";
}

#[test]
fn test_four_to_tri() {
    assert tri::eq( four::to_trit(four::true), tri::true );
    assert tri::eq( four::to_trit(four::false), tri::false );
    assert tri::eq( four::to_trit(four::none), tri::unknown );
    log_full(core::debug, four::to_trit(four::both));
    assert tri::eq( four::to_trit(four::both), tri::unknown );
}

#[test]
fn test_four_to_bit() {
    four::all_values { |v|
        assert four::to_bit(v) == if four::is_true(v) { 1u8 } else { 0u8 };
    }
}