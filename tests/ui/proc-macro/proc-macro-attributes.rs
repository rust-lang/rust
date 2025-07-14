//@ proc-macro: derive-b.rs

#[macro_use]
extern crate derive_b;

#[B] //~ ERROR `B` is ambiguous
     //~| ERROR derive helper attribute is used before it is introduced
     //~| WARN this was previously accepted
#[C] //~ ERROR cannot find attribute `C` in this scope
#[B(D)] //~ ERROR `B` is ambiguous
        //~| ERROR derive helper attribute is used before it is introduced
        //~| WARN this was previously accepted
#[B(E = "foo")] //~ ERROR `B` is ambiguous
                //~| ERROR derive helper attribute is used before it is introduced
                //~| WARN this was previously accepted
#[B(arbitrary tokens)] //~ ERROR `B` is ambiguous
                       //~| ERROR derive helper attribute is used before it is introduced
                       //~| WARN this was previously accepted
#[derive(B)]
struct B;

fn main() {}
