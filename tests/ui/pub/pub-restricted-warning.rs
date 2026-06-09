//@ check-pass

#![allow(dead_code)]

mod outer {
    pub mod inner {
        pub(in crate::outer) struct Foo;
        pub fn bar() -> Foo {
            //~^ WARNING type `Foo` is more private than the item `outer::inner::bar` [private_interfaces]
            Foo
        }
    }

    pub mod nested {
        pub mod inner {
            pub(in crate::outer::nested) struct NestedFoo;
            pub fn bar() -> NestedFoo {
                //~^ WARNING type `NestedFoo` is more private than the item `nested::inner::bar` [private_interfaces]
                NestedFoo
            }
        }
    }
}

fn main() {}
