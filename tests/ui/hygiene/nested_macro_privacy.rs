#![feature(decl_macro)]

macro n($foo:ident, $S:ident, $i:ident, $m:ident) {
    mod $foo {
        #[derive(Default)]
        pub struct $S { $i: u32 }
        pub macro $m($e:expr) { $e.$i }
    }
}

n!(foo, S, i, m);

fn main() {
    use foo::{S, m};
    S::default().i; //~ ERROR field `i` of struct `S` is private
    m!(S::default()); // ok
}
