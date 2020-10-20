// check-pass

#![feature(pub_macro_rules)]

mod m {
    // `pub` `macro_rules` can be used earlier in item order than they are defined.
    foo!();

    pub macro_rules! foo { () => {} }

    // `pub(...)` works too.
    pub(super) macro_rules! bar { () => {} }
}

// `pub` `macro_rules` are available by module path.
m::foo!();

m::bar!();

fn main() {}
