import core::*;

use std;

import std::tri;

pure fn eq1(a: tri::t, b: tri::t) -> bool { tri::eq(a , b) }
pure fn ne1(a: tri::t, b: tri::t) -> bool { tri::ne(a , b) }

pure fn eq2(a: tri::t, b: tri::t) -> bool { eq1( a, b ) && eq1( b, a ) }

#[test]
fn test_eq2() {
    tri::all_values { |a|
        tri::all_values { |b|
            assert if a == b { eq1( a, b ) } else { ne1( a, b ) }
        }
    }
}

#[test]
fn test_tri_and_symmetry() {
    tri::all_values { |a|
        tri::all_values { |b|
            assert eq1( tri::and(a ,b), tri::and(b, a) );
        }
    }
}

#[test]
fn test_tri_or_symmetry() {
    tri::all_values { |a|
        tri::all_values { |b|
            assert eq1( tri::or(a ,b), tri::or(b, a) );
        }
    }
}

#[test]
fn test_tri_xor_symmetry() {
    tri::all_values { |a|
        tri::all_values { |b|
            assert eq1( tri::xor(a ,b), tri::xor(b, a) );
        }
    }
}

#[test]
fn test_tri_not() {
    assert eq2( tri::not(tri::true), tri::false);
    assert eq2( tri::not(tri::unknown), tri::unknown);
    assert eq2( tri::not(tri::false), tri::true);
}

#[test]
fn test_tri_and() {
    assert eq2( tri::and(tri::true, tri::true), tri::true);
    assert eq2( tri::and(tri::true, tri::false), tri::false);
    assert eq2( tri::and(tri::true, tri::unknown), tri::unknown);
    assert eq2( tri::and(tri::false, tri::false), tri::false);
    assert eq2( tri::and(tri::false, tri::unknown), tri::false);
    assert eq2( tri::and(tri::unknown, tri::unknown), tri::unknown);
}

#[test]
fn test_tri_or() {
    assert eq2( tri::or(tri::true, tri::true), tri::true);
    assert eq2( tri::or(tri::true, tri::false), tri::true);
    assert eq2( tri::or(tri::true, tri::unknown), tri::true);
    assert eq2( tri::or(tri::false, tri::false), tri::false);
    assert eq2( tri::or(tri::false, tri::unknown), tri::unknown);
    assert eq2( tri::or(tri::unknown, tri::unknown), tri::unknown);
}

#[test]
fn test_tri_xor() {
    assert eq2( tri::xor(tri::true, tri::true), tri::false);
    assert eq2( tri::xor(tri::false, tri::false), tri::false);
    assert eq2( tri::xor(tri::true, tri::false), tri::true);
    assert eq2( tri::xor(tri::true, tri::unknown), tri::unknown);
    assert eq2( tri::xor(tri::false, tri::unknown), tri::unknown);
    assert eq2( tri::xor(tri::unknown, tri::unknown), tri::unknown);
}

#[test]
fn test_tri_implies() {
    assert eq2( tri::implies(tri::false, tri::false), tri::true);
    assert eq2( tri::implies(tri::false, tri::unknown), tri::true);
    assert eq2( tri::implies(tri::false, tri::true), tri::true);

    assert eq2( tri::implies(tri::unknown, tri::false), tri::unknown);
    assert eq2( tri::implies(tri::unknown, tri::unknown), tri::unknown);
    assert eq2( tri::implies(tri::unknown, tri::true), tri::true);

    assert eq2( tri::implies(tri::true, tri::false), tri::false);
    assert eq2( tri::implies(tri::true, tri::unknown), tri::unknown);
    assert eq2( tri::implies(tri::true, tri::true), tri::true);
}

#[test]
fn test_tri_from_str() {
    tri::all_values { |v|
        assert eq2( v, tri::from_str(tri::to_str(v)));
    }
}

#[test]
fn test_tri_to_str() {
    assert tri::to_str(tri::false) == "false";
    assert tri::to_str(tri::unknown) == "unknown";
    assert tri::to_str(tri::true) == "true";
}

#[test]
fn test_tri_to_bit() {
    tri::all_values { |v|
        assert tri::to_bit(v) == if tri::is_true(v) { 1u8 } else { 0u8 };
    }
}