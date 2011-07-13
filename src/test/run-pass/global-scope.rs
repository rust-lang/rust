// xfail-stage0
// xfail-fast

fn f() -> int { ret 1; }

mod foo {
    fn f() -> int { ret 2; }
    fn g() {
        assert (f() == 2);
        assert (::f() == 1);
    }
}

fn main() { ret foo::g(); }

