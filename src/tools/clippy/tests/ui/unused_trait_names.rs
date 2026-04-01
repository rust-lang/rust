//@aux-build:proc_macros.rs

#![allow(unused)]
#![warn(clippy::unused_trait_names)]
#![feature(decl_macro)]

extern crate proc_macros;

fn main() {}

fn bad() {
    use std::any::Any;
    //~^ unused_trait_names

    println!("{:?}", "foo".type_id());
}

fn good() {
    use std::any::Any as _;

    println!("{:?}", "foo".type_id());
}

fn used_good() {
    use std::any::Any;

    println!("{:?}", Any::type_id("foo"));
    println!("{:?}", "foo".type_id());
}

fn multi_bad() {
    use std::any::{self, Any, TypeId};
    //~^ unused_trait_names

    println!("{:?}", "foo".type_id());
}

fn multi_good() {
    use std::any::{self, Any as _, TypeId};

    println!("{:?}", "foo".type_id());
}

fn renamed_bad() {
    use std::any::Any as MyAny;
    //~^ unused_trait_names

    println!("{:?}", "foo".type_id());
}

fn multi_renamed_bad() {
    use std::any::{Any as MyAny, TypeId as MyTypeId};
    //~^ unused_trait_names

    println!("{:?}", "foo".type_id());
}

mod pub_good {
    pub use std::any::Any;

    fn foo() {
        println!("{:?}", "foo".type_id());
    }
}

mod used_mod_good {
    use std::any::Any;

    fn foo() {
        println!("{:?}", Any::type_id("foo"));
    }
}

mod mod_import_bad {
    fn mod_import_bad() {
        use std::any::Any;
        //~^ unused_trait_names

        println!("{:?}", "foo".type_id());
    }
}

mod nested_mod_used_good1 {
    use std::any::Any;

    mod foo {
        fn foo() {
            super::Any::type_id("foo");
        }
    }
}

mod nested_mod_used_good2 {
    use std::any::Any;

    mod foo {
        use super::Any;

        fn foo() {
            Any::type_id("foo");
        }
    }
}

mod nested_mod_used_good3 {
    use std::any::Any;

    mod foo {
        use crate::nested_mod_used_good3::Any;

        fn foo() {
            println!("{:?}", Any::type_id("foo"));
        }
    }
}

mod nested_mod_used_bad {
    use std::any::Any;
    //~^ unused_trait_names

    fn bar() {
        println!("{:?}", "foo".type_id());
    }

    mod foo {
        use std::any::Any;

        fn foo() {
            println!("{:?}", Any::type_id("foo"));
        }
    }
}

// More complex example where `use std::any::Any;` should be anonymised but `use std::any::Any as
// MyAny;` should not as it is used by a sub module. Even though if you removed `use std::any::Any;`
// the code would still compile.
mod nested_mod_used_bad1 {
    use std::any::Any;
    //~^ unused_trait_names

    use std::any::Any as MyAny;

    fn baz() {
        println!("{:?}", "baz".type_id());
    }

    mod foo {
        use crate::nested_mod_used_bad1::MyAny;

        fn foo() {
            println!("{:?}", MyAny::type_id("foo"));
        }
    }
}

// Example of nested import with an unused import to try and trick it
mod nested_mod_used_good5 {
    use std::any::Any;

    mod foo {
        use std::any::Any;

        fn baz() {
            println!("{:?}", "baz".type_id());
        }

        mod bar {
            use crate::nested_mod_used_good5::foo::Any;

            fn foo() {
                println!("{:?}", Any::type_id("foo"));
            }
        }
    }
}

mod simple_trait {
    pub trait MyTrait {
        fn do_things(&self);
    }

    pub struct MyStruct;

    impl MyTrait for MyStruct {
        fn do_things(&self) {}
    }
}

// Underscore imports were stabilized in 1.33
#[clippy::msrv = "1.32"]
fn msrv_1_32() {
    use simple_trait::{MyStruct, MyTrait};
    MyStruct.do_things();
}

#[clippy::msrv = "1.33"]
fn msrv_1_33() {
    use simple_trait::{MyStruct, MyTrait};
    //~^ unused_trait_names
    MyStruct.do_things();
}

// Linting inside macro expansion is no longer supported
mod lint_inside_macro_expansion_bad {
    macro_rules! foo {
        () => {
            use std::any::Any;
            fn bar() {
                "bar".type_id();
            }
        };
    }

    foo!();
}

mod macro_and_trait_same_name {
    pub macro Foo() {}
    pub trait Foo {
        fn bar(&self);
    }
    impl Foo for () {
        fn bar(&self) {}
    }
}

fn call_macro_and_trait_good() {
    // importing trait and macro but only using macro by path won't allow us to change this to
    // `use macro_and_trait_same_name::Foo as _;`
    use macro_and_trait_same_name::Foo;
    Foo!();
    ().bar();
}

proc_macros::external!(
    fn ignore_inside_external_proc_macro() {
        use std::any::Any;
        "foo".type_id();
    }
);

proc_macros::with_span!(
    span

    fn ignore_inside_with_span_proc_macro() {
        use std::any::Any;
        "foo".type_id();
    }
);

// This should warn the import is unused but should not trigger unused_trait_names
#[warn(unused)]
mod unused_import {
    use std::any::Any;
    //~^ ERROR: unused import
}

#[allow(clippy::unused_trait_names)]
fn allow_lint_fn() {
    use std::any::Any;

    "foo".type_id();
}

#[allow(clippy::unused_trait_names)]
mod allow_lint_mod {
    use std::any::Any;

    fn foo() {
        "foo".type_id();
    }
}

mod allow_lint_import {
    #[allow(clippy::unused_trait_names)]
    use std::any::Any;

    fn foo() {
        "foo".type_id();
    }
}

// Limitation: Suggests `use std::any::Any as _::{self};` which looks weird
// fn use_trait_self_good() {
//     use std::any::Any::{self};
//     "foo".type_id();
// }

// Limitation: Suggests `use std::any::{Any as _, Any as _};`
// mod repeated_renamed {
//     use std::any::{Any, Any as MyAny};

//     fn foo() {
//         "foo".type_id();
//     }
// }
