// xfail-fast
// aux-build:crateresolve3-1.rs
// aux-build:crateresolve3-2.rs

// verify able to link with crates with same name but different versions
// as long as no name collision on invoked functions.

mod a {
    extern mod crateresolve3(vers = "0.1");
    fn f() { assert crateresolve3::f() == 10; }
}

mod b {
    extern mod crateresolve3(vers = "0.2");
    fn f() { assert crateresolve3::g() == 20; }
}

fn main() {
    a::f();
    b::f();
}
