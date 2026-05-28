#![feature(default_struct_values)]

// Test for now that nightly default field values are left alone for now.

struct Foo {
    default_field:    Spacing =    /* uwu */ 0,
}

struct Foo2 {
    #[rustfmt::skip]
    default_field:    Spacing =    /* uwu */ 0,
}

a_macro!(
    struct Foo2 {
        default_field:    Spacing =    /* uwu */ 0,
    }
);
