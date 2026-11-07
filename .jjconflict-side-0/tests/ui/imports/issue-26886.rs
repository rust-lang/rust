//@ dont-require-annotations: NOTE

use std::sync::{self, Arc};
use std::sync::Arc; //~ ERROR the name `Arc` is defined multiple times
                    //~| NOTE `Arc` must be defined only once in the type namespace of this module
use std::sync; //~ ERROR the name `sync` is defined multiple times
               //~| NOTE `sync` must be defined only once in the type namespace of this module

fn main() {
}
