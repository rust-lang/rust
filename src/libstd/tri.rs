// -*- rust -*-

/*
Module: tri

ADT for the ternary Kleene logic K3

This allows reasoning with three logic values (true, false, unknown).

Implementation: Truth values are represented using a single u8 and
all operations are done using bit operations which is fast
on current cpus.
*/

export t, true, false, unknown;
export not, and, or, xor, implies, eq, ne, is_true, is_false;
export from_str, to_str, all_values, to_bit;

/*
Type: t

The type of ternary logic values
*/
type t = u8;

const b0: u8  = 1u8;
const b1: u8  = 2u8;
const b01: u8 = 3u8;

/*
Constant: unknown

Logic value for unknown (maybe true xor maybe false)
*/
const unknown: t = 0u8;

/*
Constant: true

Logic value for truth
*/
const true: t = 1u8;

/*
Constant: false

Logic value for falsehood
*/
const false: t = 2u8;

/* Function: not

Negation/Inverse
*/
pure fn not(v: t) -> t { ((v << 1u8) | (v >> 1u8)) & b01 }

/* Function: and

Conjunction
*/
pure fn and(a: t, b: t) -> t { ((a | b) & b1) | ((a & b) & b0) }

/* Function: or

Disjunction
*/
pure fn or(a: t, b: t) -> t { ((a & b) & b1) | ((a | b) & b0) }

/*
Function: xor

Exclusive or
*/
pure fn xor(a: t, b: t) -> t {
    let anb = a & b;
    let aob = a & not(b);
    ret ((anb & b1) | (anb << 1u8) | (aob >> 1u8) | (aob & b0)) & b01;
}

/*
Function: implies

Classic implication, i.e. from `a` follows `b`
*/
pure fn implies(a: t, b: t) -> t {
    ret ((a & b1) >> 1u8) | (b & b0) | ((a << 1u8) & b & b1);
}

/*
Predicate: eq

Returns:

true if truth values `a` and `b` are indistinguishable in the logic
*/
pure fn eq(a: t, b: t) -> bool {  a == b }

/*
Predicate: ne

Returns:

true if truth values `a` and `b` are distinguishable in the logic
*/
pure fn ne(a: t, b: t) -> bool { a != b }

/*
Predicate: is_true

Returns:

true if `v` represents truth in the logic
*/
pure fn is_true(v: t) -> bool { v == tri::true }

/*
Predicate: is_false

Returns:

true if `v` represents false in the logic
*/
pure fn is_false(v: t) -> bool { v == tri::false }

/*
Predicate: is_unknown

Returns:

true if `v` represents the unknown state in the logic
*/
pure fn is_unknown(v: t) -> bool { v == unknown }

/*
Function: from_str

Parse logic value from `s`
*/
pure fn from_str(s: str) -> t {
    alt s {
      "unknown" { unknown }
      "true" { tri::true }
      "false" { tri::false }
    }
}

/*
Function: to_str

Convert `v` into a string
*/
pure fn to_str(v: t) -> str {
    // FIXME replace with consts as soon as that works
    alt v {
      0u8 { "unknown" }
      1u8 { "true" }
      2u8 { "false" }
    }
}

/*
Function: all_values

Iterates over all truth values by passing them to `blk`
in an unspecified order
*/
fn all_values(blk: fn(v: t)) {
    blk(tri::false);
    blk(unknown);
    blk(tri::true);
}

/*
Function: to_bit

Returns:

An u8 whose first bit is set if `if_true(v)` holds
*/
fn to_bit(v: t) -> u8 { v & b0 }

#[cfg(test)]
mod tests {

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
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
