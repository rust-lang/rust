#![allow(bare_trait_objects)]
struct Struct;
trait Trait {}
impl Trait for Struct {}
impl Trait for u32 {}

fn fuz() -> (usize, Trait) { (42, Struct) }
//~^ ERROR the size for values of type `(dyn Trait + 'static)` cannot be known at compilation time
//~| ERROR the size for values of type `(dyn Trait + 'static)` cannot be known at compilation time
//~| ERROR mismatched types

fn bar() -> (usize, dyn Trait) { (42, Struct) }
//~^ ERROR the size for values of type `(dyn Trait + 'static)` cannot be known at compilation time
//~| ERROR the size for values of type `(dyn Trait + 'static)` cannot be known at compilation time
//~| ERROR mismatched types

fn bap() -> Trait { Struct }
//~^ ERROR return type cannot be a trait object without pointer indirection
//~| ERROR return type cannot be a trait object without pointer indirection

fn ban() -> dyn Trait { Struct }
//~^ ERROR return type cannot be a trait object without pointer indirection
//~| ERROR return type cannot be a trait object without pointer indirection

fn bak() -> dyn Trait { unimplemented!() }
//~^ ERROR return type cannot be a trait object without pointer indirection
//~| ERROR return type cannot be a trait object without pointer indirection

fn bal() -> dyn Trait {
    //~^ ERROR return type cannot be a trait object without pointer indirection
    //~| ERROR return type cannot be a trait object without pointer indirection
    if true {
        return Struct;
    }
    42
}

fn bax() -> dyn Trait {
    //~^ ERROR return type cannot be a trait object without pointer indirection
    //~| ERROR return type cannot be a trait object without pointer indirection
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
fn bat() -> dyn Trait {
    //~^ ERROR return type cannot be a trait object without pointer indirection
    //~| ERROR return type cannot be a trait object without pointer indirection
    if true {
        return 0;
    }
    42
}

fn bay() -> dyn Trait {
    //~^ ERROR return type cannot be a trait object without pointer indirection
    //~| ERROR return type cannot be a trait object without pointer indirection
    if true {
        0
    } else {
        42
    }
}

fn main() {}
