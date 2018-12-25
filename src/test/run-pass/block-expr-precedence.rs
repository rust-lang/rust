#![allow(unused_must_use)]
#![allow(unused_parens)]
// This test has some extra semis in it that the pretty-printer won't
// reproduce so we don't want to automatically reformat it

// no-reformat


/*
 *
 *  When you write a block-expression thing followed by
 *  a lone unary operator, you can get a surprising parse:
 *
 *  if (...) { ... }
 *  -num;
 *
 * for example, or:
 *
 *  if (...) { ... }
 *  *box;
 *
 * These will parse as subtraction and multiplication binops.
 * To get them to parse "the way you want" you need to brace
 * the leading unops:

 *  if (...) { ... }
 *  {-num};
 *
 * or alternatively, semi-separate them:
 *
 *  if (...) { ... };
 *  -num;
 *
 * This seems a little wonky, but the alternative is to lower
 * precedence of such block-like exprs to the point where
 * you have to parenthesize them to get them to occur in the
 * RHS of a binop. For example, you'd have to write:
 *
 *   12 + (if (foo) { 13 } else { 14 });
 *
 * rather than:
 *
 *   12 + if (foo) { 13 } else { 14 };
 *
 * Since we want to maintain the ability to write the latter,
 * we leave the parens-burden on the trailing unop case.
 *
 */

pub fn main() {

  let num = 12;

  assert_eq!(if (true) { 12 } else { 12 } - num, 0);
  assert_eq!(12 - if (true) { 12 } else { 12 }, 0);
  if (true) { 12; } {-num};
  if (true) { 12; }; {-num};
  if (true) { 12; };;; -num;
}
