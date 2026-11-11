#![crate_type = "lib"]

struct Bug([u8; panic!{"\t"}]);
//~^ ERROR evaluation panicked
//~| NOTE: in this expansion of panic!
