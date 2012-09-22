// We can export tags without exporting the variants to create a simple
// sort of ADT.

mod foo {
    #[legacy_exports];
    export t;
    export f;

    enum t { t1, }

    fn f() -> t { return t1; }
}

fn main() { let v: foo::t = foo::f(); }
