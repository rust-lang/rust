#[link(name = "crate_method_reexport_grrrrrrr2")];

export rust;

use name_pool::add;

mod name_pool {
    #[legacy_exports];

    type name_pool = ();

    trait add {
        fn add(s: ~str);
    }

    impl name_pool: add {
        fn add(s: ~str) {
        }
    }
}

mod rust {
    #[legacy_exports];

    use name_pool::add;
    export add;
    export rt;
    export cx;

    type rt = @();

    trait cx {
        fn cx();
    }

    impl rt: cx {
        fn cx() {
        }
    }
}
