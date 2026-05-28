// Regression test for ICE #125876
//@ edition: 2015

fn main() {
    std::ptr::from_ref(num).cast_mut().as_deref();
    //~^ ERROR cannot find value `num` in this scope
    //~| ERROR no method named `as_deref` found for raw pointer `*mut _` in the current scope
    //~| WARN type annotations needed
    //~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2018!
    //~| WARN type annotations needed
    //~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2018!
}
