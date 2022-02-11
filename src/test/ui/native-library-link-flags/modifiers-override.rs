// compile-flags:-ldylib:+as-needed=foo -lstatic=bar -Zunstable-options

#![feature(native_link_modifiers_bundle)]

#[link(name = "foo")]
#[link( //~ ERROR multiple `modifiers` arguments in a single `#[link]` attribute
    name = "bar",
    kind = "static",
    modifiers = "+whole-archive,-whole-archive",
    //~^ ERROR same modifier is used multiple times in a single `modifiers` argument
    modifiers = "+bundle"
)]
extern "C" {}
//~^ ERROR overriding linking modifiers from command line is not supported
//~| ERROR overriding linking modifiers from command line is not supported

fn main() {}
