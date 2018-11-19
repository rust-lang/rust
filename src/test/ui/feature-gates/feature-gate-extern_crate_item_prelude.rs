// edition:2018

#![feature(alloc, underscore_imports)]

extern crate alloc;

mod in_scope {
    fn check() {
        let v = alloc::vec![0];
        //~^ ERROR use of extern prelude names introduced with `extern crate` items is unstable
        type A = alloc::boxed::Box<u8>;
        //~^ ERROR use of extern prelude names introduced with `extern crate` items is unstable
    }
}

mod absolute {
    fn check() {
        let v = ::alloc::vec![0];
        //~^ ERROR use of extern prelude names introduced with `extern crate` items is unstable
        type A = ::alloc::boxed::Box<u8>;
        //~^ ERROR use of extern prelude names introduced with `extern crate` items is unstable
    }
}

mod import_in_scope {
    use alloc as _;
    //~^ ERROR use of extern prelude names introduced with `extern crate` items is unstable
    use alloc::boxed;
    //~^ ERROR use of extern prelude names introduced with `extern crate` items is unstable
}

mod import_absolute {
    use ::alloc;
    //~^ ERROR use of extern prelude names introduced with `extern crate` items is unstable
    use ::alloc::boxed;
    //~^ ERROR use of extern prelude names introduced with `extern crate` items is unstable
}

extern crate alloc as core;

mod unrelated_crate_renamed {
    type A = core::boxed::Box<u8>;
    //~^ ERROR use of extern prelude names introduced with `extern crate` items is unstable
}

fn main() {}
