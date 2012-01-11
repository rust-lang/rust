// -*- rust -*-

#[doc = "Classic Boolean logic reified as ADT"];

export t;
export not, and, or, xor, implies;
export eq, ne, is_true, is_false;
export from_str, to_str, all_values, to_bit;

#[doc = "The type of boolean logic values"]
type t = bool;

#[doc(
  brief = "Negation/Inverse",
  args(v = "Value to Negate/Invert"),
  return = "Negated/Inverted Value"
)]
pure fn not(v: t) -> t { !v }

#[doc(
  brief = "Conjunction",
  args(a = "value `a`",
       b = "value `b`"),
  return = "`a` AND `b`"
)]
pure fn and(a: t, b: t) -> t { a && b }

#[doc(
  brief = "Disjunction",
  args(a = "value `a`",
       b = "value `b`"),
  return = "`a` OR `b`"
)]
pure fn or(a: t, b: t) -> t { a || b }

#[doc(
  brief = "Exclusive or, i.e. `or(and(a, not(b)), and(not(a), b))`",
  args(a = "value `a`",
       b = "value `b`"),
  return = "`a` XOR `b`"
)]
pure fn xor(a: t, b: t) -> t { (a && !b) || (!a && b) }

#[doc(
  brief = "Implication in the logic, i.e. from `a` follows `b`",
  args(a = "value `a`",
       b = "value `b`"),
  return = "`a` IMPLIES `b`"
)]
pure fn implies(a: t, b: t) -> t { !a || b }

#[doc(
  brief = "true if truth values `a` and `b` are indistinguishable in the logic",
  args(a = "value `a`",
       b = "value `b`"),
  return = "`a` == `b`"
)]
pure fn eq(a: t, b: t) -> bool { a == b }

#[doc(
  brief = "true if truth values `a` and `b` are distinguishable in the logic",
  args(a = "value `a`",
       b = "value `b`"),
  return = "`a` != `b`"
)]
pure fn ne(a: t, b: t) -> bool { a != b }

#[doc(
  brief = "true if `v` represents truth in the logic",
  args(v = "value `v`"),
  return = "bool(`v`)"
)]
pure fn is_true(v: t) -> bool { v }

#[doc(
  brief = "true if `v` represents falsehood in the logic",
  args(v = "value `v`"),
  return = "bool(!`v`)"
)]
pure fn is_false(v: t) -> bool { !v }

#[doc(
  brief = "Parse logic value from `s`",
  args(v = "string value `s`"),
  return = "true if `s` equals \"true\", else false"
)]
pure fn from_str(s: str) -> t {
    alt s {
      "true" { true }
      "false" { false }
    }
}

#[doc(
  brief = "Convert `v` into a string",
  args(v = "truth value `v`"),
  return = "\"true\" if value `v` is true, else \"false\""
)]
pure fn to_str(v: t) -> str { if v { "true" } else { "false" } }

#[doc(
  brief = "Iterates over all truth values by passing them to `blk` in an unspecified order",
  args(v = "block value `v`"),
  return = "Undefined return value"
)]
fn all_values(blk: block(v: t)) {
    blk(true);
    blk(false);
}

#[doc(
  brief = "converts truth value to an 8 bit byte",
  args(v = "value `v`"),
  return = "returns byte with value 1 if `v` has truth value of true, else 0"
)]
pure fn to_bit(v: t) -> u8 { if v { 1u8 } else { 0u8 } }

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
