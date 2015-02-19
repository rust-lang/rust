// aux-build:struct-field-privacy.rs

extern crate "struct-field-privacy" as xc;

use xc::B;

fn main () {
    let k = B {
        aa: 20, //~ ERROR structure `struct-field-privacy::B` has no field named `aa`
        //~^ HELP did you mean `a`?
        bb: 20, //~ ERROR structure `struct-field-privacy::B` has no field named `bb`
    };
}
