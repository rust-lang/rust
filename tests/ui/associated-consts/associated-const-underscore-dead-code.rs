#![feature(associated_const_underscore)]
#![deny(dead_code)]

fn main() {}

struct Struct {}

impl Struct {
    const _: () = {
        struct Unused;
        //~^ ERROR: struct `Unused` is never constructed [dead_code]
    };
}
