// xfail-fast

#[legacy_exports];

fn f() -> int { return 1; }

mod foo {
    #[legacy_exports];
    fn f() -> int { return 2; }
    fn g() { assert (f() == 2); assert (::f() == 1); }
}

fn main() { return foo::g(); }

