#[link(name = "crate_method_reexport_grrrrrrr2")];

export rust;

import name_pool::add;
import name_pool::methods;

mod name_pool {

    type name_pool = ();

    trait add {
        fn add(s: ~str);
    }

    impl methods of add for name_pool {
        fn add(s: ~str) {
        }
    }
}

mod rust {

    import name_pool::add;
    export add;
    export rt;
    export methods;
    export cx;

    type rt = @();

    trait cx {
        fn cx();
    }

    impl methods of cx for rt {
        fn cx() {
        }
    }
}
