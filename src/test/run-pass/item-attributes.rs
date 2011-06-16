// xfail-stage0

// These are are attributes of the following mod
#[attr1 = "val"]
#[attr2 = "val"]
mod test_first_item_in_file_mod {
}

mod test_single_attr_outer {

    #[attr = "val"]
    const int x = 10;

    #[attr = "val"]
    fn f() {}

    #[attr = "val"]
    mod mod1 {
    }

    #[attr = "val"]
    native "rust" mod rustrt { }

    #[attr = "val"]
    type t = obj { };


    #[attr = "val"]
    obj o() { }
}

mod test_multi_attr_outer {

    #[attr1 = "val"]
    #[attr2 = "val"]
    const int x = 10;

    #[attr1 = "val"]
    #[attr2 = "val"]
    fn f() {}

    #[attr1 = "val"]
    #[attr2 = "val"]
    mod mod1 {
    }

    #[attr1 = "val"]
    #[attr2 = "val"]
    native "rust" mod rustrt { }

    #[attr1 = "val"]
    #[attr2 = "val"]
    type t = obj { };


    #[attr1 = "val"]
    #[attr2 = "val"]
    obj o() { }
}

mod test_stmt_single_attr_outer {

    fn f() {

        #[attr = "val"]
        const int x = 10;

        #[attr = "val"]
        fn f() {}

        /* FIXME: Issue #493
        #[attr = "val"]
        mod mod1 {
        }

        #[attr = "val"]
        native "rust" mod rustrt {
        }
        */

        #[attr = "val"]
        type t = obj { };

        #[attr = "val"]
        obj o() { }

    }
}

mod test_stmt_multi_attr_outer {

    fn f() {

        #[attr1 = "val"]
        #[attr2 = "val"]
        const int x = 10;

        #[attr1 = "val"]
        #[attr2 = "val"]
        fn f() {}

        /* FIXME: Issue #493
        #[attr1 = "val"]
        #[attr2 = "val"]
        mod mod1 {
        }

        #[attr1 = "val"]
        #[attr2 = "val"]
        native "rust" mod rustrt {
        }
        */

        #[attr1 = "val"]
        #[attr2 = "val"]
        type t = obj { };

        #[attr1 = "val"]
        #[attr2 = "val"]
        obj o() { }

    }
}

mod test_attr_inner {

    mod m {
        // This is an attribute of mod m
        #[attr = "val"];
    }
}

mod test_attr_inner_then_outer {

    mod m {
        // This is an attribute of mod m
        #[attr = "val"];
        // This is an attribute of fn f
        #[attr = "val"]
        fn f() {
        }
    }
}

mod test_attr_inner_then_outer_multi {
    mod m {
        // This is an attribute of mod m
        #[attr1 = "val"];
        #[attr2 = "val"];
        // This is an attribute of fn f
        #[attr1 = "val"]
        #[attr2 = "val"]
        fn f() {
        }
    }
}

fn main() {
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
