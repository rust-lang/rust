// The trait must not be available if its feature flag is absent.

#![crate_type = "lib"]

use std::mem::BikeshedIntrinsicFrom;
//~^ ERROR use of unstable library feature 'transmutability' [E0658]
