// run-pass
// ignore-pretty pretty-printing is unhygienic

#![feature(decl_macro)]

macro m($T:ident) {
    fn f<T, $T>(t: T, t2: $T) -> (T, $T) {
        (t, t2)
    }
}

m!(T);

fn main() {}
