// These are attributes of the implicit crate. Really this just needs to parse
// for completeness since .rs files linked from .rc files support this
// notation to specify their module's attributes

// check-pass

#![feature(rustc_attrs)]
#![rustc_dummy = "val"]
#![rustc_dummy = "val"]
#![rustc_dummy]
#![rustc_dummy(attr5)]

// These are attributes of the following mod
#[rustc_dummy = "val"]
#[rustc_dummy = "val"]
mod test_first_item_in_file_mod {}

mod test_single_attr_outer {
    #[rustc_dummy = "val"]
    pub static X: isize = 10;

    #[rustc_dummy = "val"]
    pub fn f() {}

    #[rustc_dummy = "val"]
    pub mod mod1 {}

    pub mod rustrt {
        #[rustc_dummy = "val"]
        extern "C" {}
    }
}

mod test_multi_attr_outer {
    #[rustc_dummy = "val"]
    #[rustc_dummy = "val"]
    pub static X: isize = 10;

    #[rustc_dummy = "val"]
    #[rustc_dummy = "val"]
    pub fn f() {}

    #[rustc_dummy = "val"]
    #[rustc_dummy = "val"]
    pub mod mod1 {}

    pub mod rustrt {
        #[rustc_dummy = "val"]
        #[rustc_dummy = "val"]
        extern "C" {}
    }

    #[rustc_dummy = "val"]
    #[rustc_dummy = "val"]
    struct T {
        x: isize,
    }
}

mod test_stmt_single_attr_outer {
    pub fn f() {
        #[rustc_dummy = "val"]
        static X: isize = 10;

        #[rustc_dummy = "val"]
        fn f() {}

        #[rustc_dummy = "val"]
        mod mod1 {}

        mod rustrt {
            #[rustc_dummy = "val"]
            extern "C" {}
        }
    }
}

mod test_stmt_multi_attr_outer {
    pub fn f() {
        #[rustc_dummy = "val"]
        #[rustc_dummy = "val"]
        static X: isize = 10;

        #[rustc_dummy = "val"]
        #[rustc_dummy = "val"]
        fn f() {}

        #[rustc_dummy = "val"]
        #[rustc_dummy = "val"]
        mod mod1 {}

        mod rustrt {
            #[rustc_dummy = "val"]
            #[rustc_dummy = "val"]
            extern "C" {}
        }
    }
}

mod test_attr_inner {
    pub mod m {
        // This is an attribute of mod m
        #![rustc_dummy = "val"]
    }
}

mod test_attr_inner_then_outer {
    pub mod m {
        // This is an attribute of mod m
        #![rustc_dummy = "val"]
        // This is an attribute of fn f
        #[rustc_dummy = "val"]
        fn f() {}
    }
}

mod test_attr_inner_then_outer_multi {
    pub mod m {
        // This is an attribute of mod m
        #![rustc_dummy = "val"]
        #![rustc_dummy = "val"]
        // This is an attribute of fn f
        #[rustc_dummy = "val"]
        #[rustc_dummy = "val"]
        fn f() {}
    }
}

mod test_distinguish_syntax_ext {
    pub fn f() {
        format!("test{}", "s");
        #[rustc_dummy = "val"]
        fn g() {}
    }
}

mod test_other_forms {
    #[rustc_dummy]
    #[rustc_dummy(word)]
    #[rustc_dummy(attr(word))]
    #[rustc_dummy(key1 = "val", key2 = "val", attr)]
    pub fn f() {}
}

mod test_foreign_items {
    pub mod rustrt {
        extern "C" {
            #![rustc_dummy]

            #[rustc_dummy]
            fn rust_get_test_int() -> u32;
        }
    }
}

// FIXME(#623): - these aren't supported yet
/*mod test_literals {
    #![str = "s"]
    #![char = 'c']
    #![isize = 100]
    #![usize = 100_usize]
    #![mach_int = 100u32]
    #![float = 1.0]
    #![mach_float = 1.0f32]
    #![nil = ()]
    #![bool = true]
    mod m {}
}*/

fn test_fn_inner() {
    #![rustc_dummy]
}

fn main() {}
