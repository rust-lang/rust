// Using labeled break in a while loop has caused an illegal instruction being
// generated, and an ICE later.
//
// See https://github.com/rust-lang/rust/issues/51350 for more information.
//
// It is now forbidden by the HIR const-checker.
//
// See https://github.com/rust-lang/rust/pull/66170.

const CRASH: () = 'a: while break 'a {}; //~ ERROR `while` is not allowed in a `const`

fn main() {}
