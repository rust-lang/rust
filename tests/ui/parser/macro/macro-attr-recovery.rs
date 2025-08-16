#![crate_type = "lib"]
#![feature(macro_attr)]

macro_rules! attr {
    attr[$($args:tt)*] { $($body:tt)* } => {
        //~^ ERROR: `attr` rule argument matchers require parentheses
        //~v ERROR: attr:
        compile_error!(concat!(
            "attr: args=\"",
            stringify!($($args)*),
            "\" body=\"",
            stringify!($($body)*),
            "\"",
        ));
    };
}

#[attr]
struct S;
