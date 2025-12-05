//@ run-pass
//@ edition:2021

enum Variant {
    A,
    B, //~ WARNING: variant `B` is never constructed
}

struct A {
    field: Variant,
}

fn discriminant_is_a_ref() {
    let here = A { field: Variant::A };
    let out_ref = &here.field;

    || match out_ref { //~ WARNING: unused closure that must be used
        Variant::A => (),
        Variant::B => (),
    };
}

fn discriminant_is_a_field() {
    let here = A { field: Variant::A };

    || match here.field { //~ WARNING: unused closure that must be used
        Variant::A => (),
        Variant::B => (),
    };
}

fn main() {
    discriminant_is_a_ref();
    discriminant_is_a_field();
}
