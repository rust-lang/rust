#![feature(crate_local_distributed_slice)]

struct X;

impl X {
    #[distributed_slice(crate)]
    const A: [i32; _];
    //~^ ERROR expected this to be a module-level const or a static
}


trait Y {
    #[distributed_slice(crate)]
    const A: [i32; _];
    //~^ ERROR expected this to be a module-level const or a static
}

extern "C" {
    #[distributed_slice(crate)]
    static A: [i32; _];
    //~^ ERROR expected this to be a non-extern const or a static
}

fn main() {}
