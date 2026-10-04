//@ check-pass

#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

mod union {
    type union = i32;

    pub struct Bar {
        pub union: union,
    }

    pub fn union() -> Bar {
        Bar {
            union: 5
        }
    }
}

mod struct_union {
    pub struct union {
        pub union: u32
    }
    static union: union = union { union: 0 };

    impl union {
        pub fn union<'union>() -> &'union union {
            &union
        }
    }
    impl union {}
    trait Foo {}
    impl Foo for union {}
    trait Bar {
        fn bar() {}
    }
    impl Bar for union {}
}

mod union_union {
    pub union union {
        pub union: u32
    }
    const union: union = union { union: 0 };
    impl union {
        pub fn union() -> union {
            union
        }
    }
}

mod trait_union {
    pub trait union {
        fn union() {}
    }
    impl union for () {}
}

macro_rules! ty {
    ($ty:ty { $($field:ident:$field_ty:ty)* }) => {};
}

fn main() {
    let union = union::union();
    let _ = union.union;
    let _ = struct_union::union::union().union;
    let union = union_union::union::union();
    let _ = unsafe { union.union };
    <() as trait_union::union>::union();
    ty!(union {});
    ty!(union { union: union });
}
