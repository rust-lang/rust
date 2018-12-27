// aux-build:pub_restricted.rs

#![allow(warnings)]
extern crate pub_restricted;

mod foo {
    pub mod bar {
        pub(super) fn f() {}
        #[derive(Default)]
        pub struct S {
            pub(super) x: i32,
        }
        impl S {
            pub(super) fn f(&self) {}
            pub(super) fn g() {}
        }
    }
    fn f() {
        use foo::bar::S;
        pub(self) use foo::bar::f; // ok
        pub(super) use foo::bar::f as g; //~ ERROR cannot be re-exported
        S::default().x; // ok
        S::default().f(); // ok
        S::g(); // ok
    }
}

fn f() {
    use foo::bar::S;
    use foo::bar::f; //~ ERROR private
    S::default().x; //~ ERROR private
    S::default().f(); //~ ERROR private
    S::g(); //~ ERROR private
}

fn main() {
    use pub_restricted::Universe;
    use pub_restricted::Crate; //~ ERROR private

    let u = Universe::default();
    let _ = u.x;
    let _ = u.y; //~ ERROR private
    let _ = u.z; //~ ERROR private
    u.f();
    u.g(); //~ ERROR private
    u.h(); //~ ERROR private
}

mod pathological {
    pub(in bad::path) mod m1 {} //~ ERROR failed to resolve: maybe a missing `extern crate bad;`?
    pub(in foo) mod m2 {} //~ ERROR visibilities can only be restricted to ancestor modules
}
