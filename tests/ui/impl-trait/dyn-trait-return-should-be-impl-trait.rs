//@revisions: edition2015 edition2021
//@[edition2015] edition:2015
//@[edition2021] edition:2021
#![allow(bare_trait_objects)]
struct Struct;
trait Trait {}
impl Trait for Struct {}
impl Trait for u32 {}

fn fuz() -> (usize, Trait) { (42, Struct) }
//[edition2015]~^ ERROR E0277
//[edition2015]~| ERROR E0277
//[edition2015]~| ERROR E0308
//[edition2021]~^^^^ ERROR expected a type, found a trait
fn bar() -> (usize, dyn Trait) { (42, Struct) }
//~^ ERROR E0277
//~| ERROR E0277
//~| ERROR E0308
fn bap() -> Trait { Struct }
//[edition2015]~^ ERROR E0746
//[edition2021]~^^ ERROR expected a type, found a trait
fn ban() -> dyn Trait { Struct }
//~^ ERROR E0746
fn bak() -> dyn Trait { unimplemented!() } //~ ERROR E0746
// Suggest using `Box<dyn Trait>`
fn bal() -> dyn Trait { //~ ERROR E0746
    if true {
        return Struct;
    }
    42
}
fn bax() -> dyn Trait { //~ ERROR E0746
    if true {
        Struct
    } else {
        42
    }
}
fn bam() -> Box<dyn Trait> {
    if true {
        return Struct; //~ ERROR mismatched types
    }
    42 //~ ERROR mismatched types
}
fn baq() -> Box<dyn Trait> {
    if true {
        return 0; //~ ERROR mismatched types
    }
    42 //~ ERROR mismatched types
}
fn baz() -> Box<dyn Trait> {
    if true {
        Struct //~ ERROR mismatched types
    } else {
        42 //~ ERROR mismatched types
    }
}
fn baw() -> Box<dyn Trait> {
    if true {
        0 //~ ERROR mismatched types
    } else {
        42 //~ ERROR mismatched types
    }
}

// Suggest using `impl Trait`
fn bat() -> dyn Trait { //~ ERROR E0746
    if true {
        return 0;
    }
    42
}
fn bay() -> dyn Trait { //~ ERROR E0746
    if true {
        0
    } else {
        42
    }
}

fn main() {}
