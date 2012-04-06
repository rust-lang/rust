// xfail-fast
// xfail-test
// aux-build:crateresolve-1.rs
// aux-build:crateresolve-2.rs
// aux-build:crateresolve-3.rs

mod a {
    use crateresolve(vers = "0.1");
    fn f() { assert crateresolve::f() == 10; }
}

mod b {
    use crateresolve(vers = "0.2");
    fn f() { assert crateresolve::f() == 20; }
}

mod c {
    use crateresolve(vers = "0.3");
    fn f() { assert crateresolve::f() == 30; }
}

fn main() {
    a::f();
    b::f();
    c::f();
}
