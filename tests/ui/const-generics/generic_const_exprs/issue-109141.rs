#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

impl EntriesBuffer {
    fn a(&self) -> impl Iterator {
        self.0.iter_mut() //~ ERROR: cannot borrow `*self.0` as mutable, as it is behind a `&` reference
                          //~| ERROR captures lifetime that does not appear in bounds
    }
}

struct EntriesBuffer(Box<[[u8; HashesEntryLEN]; 5]>);
//~^ ERROR: cannot find value `HashesEntryLEN` in this scope

fn main() {}
