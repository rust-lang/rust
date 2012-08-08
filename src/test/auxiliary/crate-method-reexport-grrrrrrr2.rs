#[link(name = "crate_method_reexport_grrrrrrr2")];

export rust;

import name_pool::add;

mod name_pool {

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

    import name_pool::add;
    // FIXME #3155: this is a hack
    import name_pool::__extensions__;
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
