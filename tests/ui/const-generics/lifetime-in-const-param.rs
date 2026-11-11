// https://github.com/rust-lang/rust/issues/113462

struct S2<'b>(&'b ());

struct S<'a, const N: S2>(&'a ());
//~^ ERROR missing lifetime specifier [E0106]

fn main() {}
