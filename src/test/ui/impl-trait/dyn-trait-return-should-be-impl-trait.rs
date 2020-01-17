#![allow(bare_trait_objects)]
struct Struct;
trait Trait {}
impl Trait for Struct {}
impl Trait for u32 {}

fn fuz() -> (usize, Trait) { (42, Struct) }
//~^ ERROR E0277
//~| ERROR E0308
fn bar() -> (usize, dyn Trait) { (42, Struct) }
//~^ ERROR E0277
//~| ERROR E0308
fn bap() -> Trait { Struct }
//~^ ERROR E0746
fn ban() -> dyn Trait { Struct }
//~^ ERROR E0746
fn bak() -> dyn Trait { unimplemented!() } //~ ERROR E0277
// Suggest using `Box<dyn Trait>`
fn bal() -> dyn Trait { //~ ERROR E0746
    if true {
        return Struct;
    }
    42
}

// Suggest using `impl Trait`
fn bat() -> dyn Trait { //~ ERROR E0746
    if true {
        return 0;
    }
    42
}

fn main() {}
