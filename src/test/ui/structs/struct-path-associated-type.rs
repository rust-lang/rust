struct S;

trait Tr {
    type A;
}

impl Tr for S {
    type A = S;
}

fn f<T: Tr>() {
    let s = T::A {};
    //~^ ERROR expected struct, variant or union type, found associated type
    let z = T::A::<u8> {};
    //~^ ERROR expected struct, variant or union type, found associated type
    //~| ERROR type arguments are not allowed for this type
    match S {
        T::A {} => {}
        //~^ ERROR expected struct, variant or union type, found associated type
    }
}

fn g<T: Tr<A = S>>() {
    let s = T::A {}; // OK
    let z = T::A::<u8> {}; //~ ERROR type arguments are not allowed for this type
    match S {
        T::A {} => {} // OK
    }
}

fn main() {
    let s = S::A {}; //~ ERROR ambiguous associated type
    let z = S::A::<u8> {}; //~ ERROR ambiguous associated type
    //~^ ERROR type arguments are not allowed for this type
    match S {
        S::A {} => {} //~ ERROR ambiguous associated type
    }
}
