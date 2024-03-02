//@ known-bug: #110395
#![feature(const_trait_impl)]

struct Bug {
    inner: [(); match || 1 {
        n => n(),
        //FIXME ~^ ERROR the trait
        //FIXME ~| ERROR the trait
        //FIXME ~| ERROR cannot call non-const closure in constants
    }],
}

fn main() {}
