use std;

import tern = std::logic::tern;
import quart = std::logic::quart;

fn tern_req_eq1(a: tern::t, b: tern:: t) { assert( tern::eq(a , b) ); }
fn tern_req_ne1(a: tern::t, b: tern:: t) { assert( tern::ne(a , b) ); }

fn tern_req_eq(a: tern::t, b: tern:: t) {
    tern_req_eq1( a, b );
    tern_req_eq1( b, a );
}

fn tern_req_ne(a: tern::t, b: tern:: t) {
    tern_req_ne1( a, b);
    tern_req_ne1( b, a );
}

#[test]
fn test_tern_req_eq() {
    tern::all_values { |a|
        tern::all_values { |b|
            if a == b { tern_req_eq1( a, b ) } else { tern_req_ne1( a, b ) }
        }
    }
}

#[test]
fn test_tern_and_symmetry() {
    tern::all_values { |a|
        tern::all_values { |b|
            tern_req_eq1( tern::and(a ,b), tern::and(b, a) );
        }
    }
}

#[test]
fn test_tern_or_symmetry() {
    tern::all_values { |a|
        tern::all_values { |b|
            tern_req_eq1( tern::or(a ,b), tern::or(b, a) );
        }
    }
}

#[test]
fn test_tern_xor_symmetry() {
    tern::all_values { |a|
        tern::all_values { |b|
            tern_req_eq1( tern::xor(a ,b), tern::xor(b, a) );
        }
    }
}

#[test]
fn test_tern_not() {
    tern_req_eq( tern::not(tern::true), tern::false );
    tern_req_eq( tern::not(tern::unknown), tern::unknown );
    tern_req_eq( tern::not(tern::false), tern::true );
}

#[test]
fn test_tern_and() {
    tern_req_eq( tern::and(tern::true, tern::true), tern::true );
    tern_req_eq( tern::and(tern::true, tern::false), tern::false );
    tern_req_eq( tern::and(tern::true, tern::unknown), tern::unknown );
    tern_req_eq( tern::and(tern::false, tern::false), tern::false );
    tern_req_eq( tern::and(tern::false, tern::unknown), tern::false );
    tern_req_eq( tern::and(tern::unknown, tern::unknown), tern::unknown );
}

#[test]
fn test_tern_or() {
    tern_req_eq( tern::or(tern::true, tern::true), tern::true );
    tern_req_eq( tern::or(tern::true, tern::false), tern::true );
    tern_req_eq( tern::or(tern::true, tern::unknown), tern::true );
    tern_req_eq( tern::or(tern::false, tern::false), tern::false );
    tern_req_eq( tern::or(tern::false, tern::unknown), tern::unknown );
    tern_req_eq( tern::or(tern::unknown, tern::unknown), tern::unknown );
}

#[test]
fn test_tern_xor() {
    tern_req_eq( tern::xor(tern::true, tern::true), tern::false );
    tern_req_eq( tern::xor(tern::false, tern::false), tern::false );
    tern_req_eq( tern::xor(tern::true, tern::false), tern::true );
    tern_req_eq( tern::xor(tern::true, tern::unknown), tern::unknown );
    tern_req_eq( tern::xor(tern::false, tern::unknown), tern::unknown );
    tern_req_eq( tern::xor(tern::unknown, tern::unknown), tern::unknown );
}

#[test]
fn test_tern_implies() {
    tern_req_eq( tern::implies(tern::false, tern::false), tern::true );
    tern_req_eq( tern::implies(tern::false, tern::unknown), tern::true );
    tern_req_eq( tern::implies(tern::false, tern::true), tern::true );

    tern_req_eq( tern::implies(tern::unknown, tern::false), tern::unknown );
    tern_req_eq( tern::implies(tern::unknown, tern::unknown), tern::unknown );
    tern_req_eq( tern::implies(tern::unknown, tern::true), tern::true );

    tern_req_eq( tern::implies(tern::true, tern::false), tern::false );
    tern_req_eq( tern::implies(tern::true, tern::unknown), tern::unknown );
    tern_req_eq( tern::implies(tern::true, tern::true), tern::true );
}

#[test]
fn test_tern_from_str() {
    tern::all_values { |v|
        tern_req_eq( v, tern::from_str(tern::to_str(v)) );
    }
}

#[test]
fn test_tern_to_str() {
    assert( tern::to_str(tern::false) == "false" );
    assert( tern::to_str(tern::unknown) == "unknown" );
    assert( tern::to_str(tern::true) == "true"  );
}

fn quart_req_eq1(a: quart::t, b: quart:: t) { assert( quart::eq(a , b) ); }
fn quart_req_ne1(a: quart::t, b: quart:: t) { assert( quart::ne(a , b) ); }

fn quart_req_eq(a: quart::t, b: quart:: t) {
    quart_req_eq1( a, b );
    quart_req_eq1( b, a );
}

fn quart_req_ne(a: quart::t, b: quart:: t) {
    quart_req_ne1( a, b );
    quart_req_ne1( b, a );
}

#[test]
fn test_quart_req_eq() {
    quart::all_values { |a|
        quart::all_values { |b|
            if a == b { quart_req_eq1( a, b ) } else { quart_req_ne1( a, b ) }
        }
    }
}

#[test]
fn test_quart_and_symmetry() {
    quart::all_values { |a|
        quart::all_values { |b|
            quart_req_eq1( quart::and(a ,b), quart::and(b, a) );
        }
    }
}

#[test]
fn test_quart_xor_symmetry() {
    quart::all_values { |a|
        quart::all_values { |b|
            quart_req_eq1( quart::and(a ,b), quart::and(b, a) );
        }
    }
}

#[test]
fn test_quart_or_symmetry() {
    quart::all_values { |a|
        quart::all_values { |b|
            quart_req_eq1( quart::or(a ,b), quart::or(b, a) );
        }
    }
}

fn to_tup(v: quart::t) -> (bool, bool) {
    alt v {
      0u8 { (false, false) }
      1u8 { (false, true) }
      2u8 { (true, false) }
      3u8 { (true, true) }
    }
}

#[test]
fn test_quart_not() {
    quart::all_values { |a|
        let (x, y) = to_tup(a);
        assert( to_tup(quart::not(a)) == (y, x) )
    };
}


#[test]
fn test_quart_and() {
    quart::all_values { |a|
        quart::all_values { |b|
            let (x1, y1) = to_tup(a);
            let (x2, y2) = to_tup(b);
            let (x3, y3) = to_tup(quart::and(a, b));

            assert( (x3, y3) == (x1 && x2, y1 || y2) );
        }
    };
}

#[test]
fn test_quart_or() {
    quart::all_values { |a|
        quart::all_values { |b|
            let (x1, y1) = to_tup(a);
            let (x2, y2) = to_tup(b);
            let (x3, y3) = to_tup(quart::or(a, b));

            assert( (x3, y3) == (x1 || x2, y1 && y2) );
        }
    };
}

#[test]
fn test_quart_implies() {
    quart::all_values { |a|
        quart::all_values { |b|
            let (x1, _) = to_tup(a);
            let (x2, y2) = to_tup(b);
            let (x3, y3) = to_tup(quart::implies(a, b));

            assert( (x3, y3) == (!x1 || x2, x1 && y2) );
        }
    };
}

#[test]
fn test_quart_is_true() {
    assert( !quart::is_true(quart::none) );
    assert( !quart::is_true(quart::false) );
    assert( quart::is_true(quart::true) );
    assert( quart::is_true(quart::both) );
}

#[test]
fn test_quart_is_false() {
    assert( quart::is_false(quart::none) );
    assert( quart::is_false(quart::false) );
    assert( !quart::is_false(quart::true) );
    assert( !quart::is_false(quart::both) );
}

#[test]
fn test_quart_from_str() {
    quart::all_values { |v|
        quart_req_eq( v, quart::from_str(quart::to_str(v)) );
    }
}

#[test]
fn test_quart_to_str() {
    assert( quart::to_str(quart::none) == "none" );
    assert( quart::to_str(quart::false) == "false" );
    assert( quart::to_str(quart::true) == "true"  );
    assert( quart::to_str(quart::both) == "both" );
}


