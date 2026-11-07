// Test that a nominal type (like `Foo<'a>`) outlives `'b` if its
// arguments (like `'a`) outlive `'b`.
//
// Rule OutlivesNominalType from RFC 1214.

//@ check-pass

mod variant_enum_region {
    struct Foo<'a> {
        x: &'a i32,
    }
    enum Bar<'a, 'b> {
        V(&'a Foo<'b>),
    }
}

mod rev_variant_enum_region {
    struct Foo<'a> {
        x: fn(&'a i32),
    }
    enum Bar<'a, 'b> {
        V(&'a Foo<'b>),
    }
}

mod variant_enum_type {
    struct Foo<T> {
        x: T,
    }
    enum Bar<'a, 'b> {
        V(&'a Foo<&'b i32>),
    }
}

mod rev_variant_enum_type {
    struct Foo<T> {
        x: fn(T),
    }
    enum Bar<'a, 'b> {
        V(&'a Foo<&'b i32>),
    }
}

mod variant_struct_region {
    struct Foo<'a> {
        x: &'a i32,
    }
    struct Bar<'a, 'b> {
        f: &'a Foo<'b>,
    }
}

mod rev_variant_struct_region {
    struct Foo<'a> {
        x: fn(&'a i32),
    }
    struct Bar<'a, 'b> {
        f: &'a Foo<'b>,
    }
}

mod variant_struct_type {
    struct Foo<T> {
        x: T,
    }
    struct Bar<'a, 'b> {
        f: &'a Foo<&'b i32>,
    }
}

mod rev_variant_struct_type {
    struct Foo<T> {
        x: fn(T),
    }
    struct Bar<'a, 'b> {
        f: &'a Foo<&'b i32>,
    }
}

fn main() {}
