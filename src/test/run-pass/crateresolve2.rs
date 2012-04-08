// xfail-fast
// aux-build:crateresolve2-1.rs
// aux-build:crateresolve2-2.rs
// aux-build:crateresolve2-3.rs

mod a {
    use crateresolve2(vers = "0.1");
    fn f() { assert crateresolve2::f() == 10; }
}

mod b {
    use crateresolve2(vers = "0.2");
    fn f() { assert crateresolve2::f() == 20; }
}

mod c {
    use crateresolve2(vers = "0.3");
    fn f() { assert crateresolve2::f() == 30; }
}

fn main() {
    a::f();
    b::f();
    c::f();
}
