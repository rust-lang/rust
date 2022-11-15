#![feature(const_trait_impl)]

struct Bug {
    inner: [(); match || 1 {
        n => n(),
        //~^ ERROR the trait bound
        //~| ERROR the trait bound
        //~| ERROR cannot call non-const closure in constants
    }],
}

fn main() {}
