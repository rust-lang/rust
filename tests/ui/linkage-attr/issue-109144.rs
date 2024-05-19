#![crate_type = "lib"]
#[link(kind = "static", modifiers = "+whole-archive,+bundle")]
//~^ ERROR `#[link]` attribute requires a `name = "string"` argument
extern  {}
