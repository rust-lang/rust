// `loop`s unconditionally-broken-from used to be allowed in constants, but are now forbidden by
// the HIR const-checker.
//
// See https://github.com/rust-lang/rust/pull/66170 and
// https://github.com/rust-lang/rust/issues/62272.

const FOO: () = loop { break; }; //~ ERROR `loop` is not allowed in a `const`

fn main() {
    [FOO; { let x; loop { x = 5; break; } x }]; //~ ERROR `loop` is not allowed in a `const`
}
