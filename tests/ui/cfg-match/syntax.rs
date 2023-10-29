#![feature(cfg_match)]

cfg_match! {
    cfg(unix) => const BAD_SINGLE_ELEMENT: () = ();,
    //~^ ERROR arms without brackets are only allowed for function
    _ => const BAD_SINGLE_ELEMENT: () = ();,
}

cfg_match! {
    cfg(unix) => fn missing_comma() {}
    _ => fn missing_comma() {}
    //~^ ERROR conditional arms with a single element must end with a com
}

cfg_match! {
    cfg(unix) {
    //~^ ERROR conditional arm must be declared with a trailing
        fn regular_arm() {}
    }
    _ => { fn regular_arm() {} }
}

cfg_match! {
    cfg(unix) => { fn wildcard() {} }
    {
    //~^ ERROR the last arm is expected to be a wildcard
        fn wildcard() {}
    }
}

fn meaningless() {
    cfg_match! {
    //~^ ERROR single arm with a single element has the same effect of a st
        cfg(feature = "foo") => fn foo() {},
    }
}

pub fn main() {}
