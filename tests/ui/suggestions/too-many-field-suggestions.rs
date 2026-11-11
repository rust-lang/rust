struct Thing {
    a0: Foo,
    a1: Foo,
    a2: Foo,
    a3: Foo,
    a4: Foo,
    a5: Foo,
    a6: Foo,
    a7: Foo,
    a8: Foo,
    a9: Foo,
}

struct Foo {
    field: Field,
}

struct Field;

impl Foo {
    fn bar(&self) {}
}

fn bar(t: Thing) {
    t.bar(); //~ ERROR no method named `bar` found for struct `Thing`
    t.field; //~ ERROR no field `field` on type `Thing`
}

fn main() {}
