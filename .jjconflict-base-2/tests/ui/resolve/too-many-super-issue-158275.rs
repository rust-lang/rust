#![crate_type = "lib"]

mod outer {
    mod inner {
        struct Example(super::super::super::Impl);
        //~^ ERROR too many leading `super` keywords within `crate::outer::inner`
    }
}

struct Impl;
