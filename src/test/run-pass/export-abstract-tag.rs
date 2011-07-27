// We can export tags without exporting the variants to create a simple
// sort of ADT.

mod foo {
    export t;
    export f;

    tag t { t1; }

    fn f() -> t { ret t1; }
}

fn main() { let v: foo::t = foo::f(); }