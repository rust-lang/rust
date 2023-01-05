// aux-build:two_macros.rs

extern crate two_macros; // two identity macros `m` and `n`

mod foo {
    pub use two_macros::n as m;
}

mod m1 {
    m!(use two_macros::*;);
    use foo::m; // This shadows the glob import
}

mod m2 {
    use two_macros::*;
    m! { //~ ERROR ambiguous
        use foo::m;
    }
}

mod m3 {
    use two_macros::m;
    fn f() {
        use two_macros::n as m; // This shadows the above import
        m!();
    }

    fn g() {
        m! { //~ ERROR ambiguous
            use two_macros::n as m;
        }
    }
}

mod m4 {
    macro_rules! m { () => {} }
    use two_macros::m;
    m!();
}

fn main() {}
