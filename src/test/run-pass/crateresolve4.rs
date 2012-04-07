// xfail-fast
// aux-build:crateresolve4a-1.rs
// aux-build:crateresolve4a-2.rs
// aux-build:crateresolve4b-1.rs
// aux-build:crateresolve4b-2.rs

mod a {
    use crateresolve4b(vers = "0.1");
    fn f() { assert crateresolve4b::f() == 20; }
}

mod b {
    use crateresolve4b(vers = "0.2");
    fn f() { assert crateresolve4b::g() == 10; }
}

fn main() {
    a::f();
    b::f();
}
