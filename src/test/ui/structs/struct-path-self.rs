struct S;

trait Tr {
    fn f() {
        let s = Self {};
        //~^ ERROR expected struct, variant or union type, found Self
        let z = Self::<u8> {};
        //~^ ERROR expected struct, variant or union type, found Self
        //~| ERROR type arguments are not allowed for this type
        match s {
            Self { .. } => {}
            //~^ ERROR expected struct, variant or union type, found Self
        }
    }
}

impl Tr for S {
    fn f() {
        let s = Self {}; // OK
        let z = Self::<u8> {}; //~ ERROR type arguments are not allowed for this type
        match s {
            Self { .. } => {} // OK
        }
    }
}

impl S {
    fn g() {
        let s = Self {}; // OK
        let z = Self::<u8> {}; //~ ERROR type arguments are not allowed for this type
        match s {
            Self { .. } => {} // OK
        }
    }
}

fn main() {}
