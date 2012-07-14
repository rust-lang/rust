#[link(name = "crate_method_reexport_grrrrrrr2")];

export rust;

import name_pool::methods;

mod name_pool {

    type name_pool = ();

    impl methods for name_pool {
        fn add(s: ~str) {
        }
    }
}

mod rust {

    export rt;
    export methods;

    type rt = @();

    impl methods for rt {
        fn cx() {
        }
    }
}
