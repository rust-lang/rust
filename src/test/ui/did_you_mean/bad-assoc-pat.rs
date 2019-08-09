fn main() {
    match 0u8 {
        [u8]::AssocItem => {}
        //~^ ERROR missing angle brackets in associated item path
        //~| ERROR no associated item named `AssocItem` found for type `[u8]` in the current scope
        (u8, u8)::AssocItem => {}
        //~^ ERROR missing angle brackets in associated item path
        //~| ERROR no associated item named `AssocItem` found for type `(u8, u8)` in the current sco
        _::AssocItem => {}
        //~^ ERROR missing angle brackets in associated item path
        //~| ERROR no associated item named `AssocItem` found for type `_` in the current scope
    }
    match &0u8 {
        &(u8,)::AssocItem => {}
        //~^ ERROR missing angle brackets in associated item path
        //~| ERROR no associated item named `AssocItem` found for type `(u8,)` in the current scope
    }
}

macro_rules! pat {
    ($ty: ty) => ($ty::AssocItem)
    //~^ ERROR missing angle brackets in associated item path
    //~| ERROR no associated item named `AssocItem` found for type `u8` in the current scope
}
macro_rules! ty {
    () => (u8)
}

fn check_macros() {
    match 0u8 {
        pat!(u8) => {}
        ty!()::AssocItem => {}
        //~^ ERROR missing angle brackets in associated item path
        //~| ERROR no associated item named `AssocItem` found for type `u8` in the current scope
    }
}
