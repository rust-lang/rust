#![feature(associated_type_defaults)]

// Associated type defaults may not be assumed inside the trait defining them.
// ie. they only resolve to `<Self as Tr>::A`, not the actual type `()`
trait Tr {
    type A = ();

    fn f(p: Self::A) {
        let () = p;
        //~^ ERROR mismatched types
        //~| NOTE expected associated type, found `()`
        //~| NOTE expected associated type `<Self as Tr>::A`
        //~| NOTE consider constraining the associated type
        //~| NOTE for more information, visit
    }
}

// An impl that doesn't override the type *can* assume the default.
impl Tr for () {
    fn f(p: Self::A) {
        let () = p;
    }
}

impl Tr for u8 {
    type A = ();

    fn f(p: Self::A) {
        let () = p;
    }
}

trait AssocConst {
    type Ty = u8;

    // Assoc. consts also cannot assume that default types hold
    const C: Self::Ty = 0u8;
    //~^ ERROR mismatched types
    //~| NOTE expected associated type, found `u8`
    //~| NOTE expected associated type `<Self as AssocConst>::Ty`
    //~| NOTE consider constraining the associated type
    //~| NOTE for more information, visit
}

// An impl can, however
impl AssocConst for () {
    const C: Self::Ty = 0u8;
}

fn main() {}
