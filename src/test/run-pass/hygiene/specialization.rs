// run-pass
// ignore-pretty pretty-printing is unhygienic

#![feature(decl_macro)]

trait Tr {
    fn f(&self) -> &'static str {
        "This shouldn't happen"
    }
}

pub macro m($t:ty) {
    impl Tr for $t {
        fn f(&self) -> &'static str {
            "Run me"
        }
    }
}

struct S;
m!(S);

fn main() {
    assert_eq!(S.f(), "Run me");
}
