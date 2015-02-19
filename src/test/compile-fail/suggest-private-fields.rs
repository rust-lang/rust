// aux-build:struct-field-privacy.rs

extern crate "struct-field-privacy" as xc;

use xc::B;

struct A {
    pub a: u32,
    b: u32,
}

fn main () {
    // external crate struct
    let k = B {
        aa: 20, //~ ERROR structure `struct-field-privacy::B` has no field named `aa`
        //~^ HELP did you mean `a`?
        bb: 20, //~ ERROR structure `struct-field-privacy::B` has no field named `bb`
    };
    // local crate struct
    let l = A {
        aa: 20, //~ ERROR structure `A` has no field named `aa`
        //~^ HELP did you mean `a`?
        bb: 20, //~ ERROR structure `A` has no field named `bb`
        //~^ HELP did you mean `b`?
    };
}
