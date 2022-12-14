#![feature(negative_impls)]
#![crate_type = "lib"]

impl !Copy for str {}
//~^ ERROR only traits defined in the current crate can be implemented

impl !Copy for fn() {}
//~^ ERROR only traits defined in the current crate can be implemented

impl !Copy for () {}
//~^ ERROR only traits defined in the current crate can be implemented
