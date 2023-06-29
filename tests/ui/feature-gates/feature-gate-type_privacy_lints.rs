// check-pass

#![warn(private_interfaces)] //~ WARN unknown lint
                             //~| WARN unknown lint
                             //~| WARN unknown lint
#![warn(private_bounds)] //~ WARN unknown lint
                         //~| WARN unknown lint
                         //~| WARN unknown lint
#![warn(unnameable_types)] //~ WARN unknown lint
                           //~| WARN unknown lint
                           //~| WARN unknown lint
fn main() {}
