// xfail-fast

fn f() -> int { return 1; }

mod foo {
    fn f() -> int { return 2; }
    fn g() { assert (f() == 2); assert (::f() == 1); }
}

fn main() { return foo::g(); }

