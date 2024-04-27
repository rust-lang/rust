#![crate_type = "lib"]

struct Bug([u8; panic!{"\t"}]);
//~^ ERROR evaluation of constant value failed
//~| NOTE: in this expansion of panic!
