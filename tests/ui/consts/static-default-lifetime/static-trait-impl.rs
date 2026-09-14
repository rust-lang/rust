//@ check-pass

trait Bar<'a> {
    const STATIC: &'a str;
}

struct A;
impl Bar<'_> for A {
    const STATIC: &str = "";
}

struct B;
impl Bar<'static> for B {
    const STATIC: &str = "";
}

struct C;
impl Bar<'_> for C {
    const STATIC: &'static str = {
        struct B;
        impl Bar<'static> for B {
            const STATIC: &str = "";
        }
        ""
    };
}

fn main() {}
