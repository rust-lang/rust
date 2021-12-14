#![feature(associated_type_defaults)]

// Associated type defaults may not be assumed inside the trait defining them.
// ie. they only resolve to `<Self as Tr>::A`, not the actual type `()`
trait Tr {
    type A = (); //~ NOTE associated type defaults can't be assumed inside the trait defining them

    fn f(p: Self::A) {
        let () = p;
        //~^ ERROR mismatched types
        //~| NOTE expected associated type, found `()`
        //~| NOTE expected associated type `<Self as Tr>::A`
        //~| NOTE this expression has type `<Self as Tr>::A`
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
    type Ty = u8; //~ NOTE associated type defaults can't be assumed inside the trait defining them

    // Assoc. consts also cannot assume that default types hold
    const C: Self::Ty = 0u8;
    //~^ ERROR mismatched types
    //~| NOTE expected associated type, found `u8`
    //~| NOTE expected associated type `<Self as AssocConst>::Ty`
}

// An impl can, however
impl AssocConst for () {
    const C: Self::Ty = 0u8;
}

fn main() {}
