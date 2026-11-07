// Check that associated type defaults are wf checked.

#![feature(associated_type_defaults)]

// Default types must always be wf
trait Tr3 {
    type Ty = Vec<[u8]>;
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
}

fn main() {}
