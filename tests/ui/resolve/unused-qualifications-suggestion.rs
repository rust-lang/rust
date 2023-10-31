// run-rustfix

#![deny(unused_qualifications)]

mod foo {
    pub fn bar() {}
}

mod baz {
    pub mod qux {
        pub fn quux() {}
    }
}

fn main() {
    use foo::bar;
    foo::bar();
    //~^ ERROR unnecessary qualification

    use baz::qux::quux;
    baz::qux::quux();
    //~^ ERROR unnecessary qualification
}
