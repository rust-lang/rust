struct S;

trait Tr {
    fn f() {
        let s = Self {};
        //~^ ERROR expected struct, variant or union type, found type parameter
        let z = Self::<u8> {};
        //~^ ERROR expected struct, variant or union type, found type parameter
        //~| ERROR type arguments are not allowed on self type
        match s {
            Self { .. } => {}
            //~^ ERROR expected struct, variant or union type, found type parameter
        }
    }
}

impl Tr for S {
    fn f() {
        let s = Self {}; // OK
        let z = Self::<u8> {}; //~ ERROR type arguments are not allowed on self type
        match s {
            Self { .. } => {} // OK
        }
    }
}

impl S {
    fn g() {
        let s = Self {}; // OK
        let z = Self::<u8> {}; //~ ERROR type arguments are not allowed on self type
        match s {
            Self { .. } => {} // OK
        }
    }
}

fn main() {}
