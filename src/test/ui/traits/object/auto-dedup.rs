// run-pass

#![allow(unused_assignments)]

// Test that duplicate auto trait bounds in trait objects don't create new types.
#[allow(unused_assignments)]
use std::marker::Send as SendAlias;

// A dummy trait for the non-auto trait.
trait Trait {}

// A dummy struct to implement `Trait` and `Send`.
struct Struct;

impl Trait for Struct {}

// These three functions should be equivalent.
fn takes_dyn_trait_send(_: Box<dyn Trait + Send>) {}
fn takes_dyn_trait_send_send(_: Box<dyn Trait + Send + Send>) {}
fn takes_dyn_trait_send_sendalias(_: Box<dyn Trait + Send + SendAlias>) {}

impl dyn Trait + Send + Send {
    fn do_nothing(&self) {}
}

fn main() {
    // 1. Moving into a variable with more `Send`s and back.
    let mut dyn_trait_send = Box::new(Struct) as Box<dyn Trait + Send>;
    let dyn_trait_send_send: Box<dyn Trait + Send + Send> = dyn_trait_send;
    dyn_trait_send = dyn_trait_send_send;

    // 2. Calling methods with different number of `Send`s.
    let dyn_trait_send = Box::new(Struct) as Box<dyn Trait + Send>;
    takes_dyn_trait_send_send(dyn_trait_send);

    let dyn_trait_send_send = Box::new(Struct) as Box<dyn Trait + Send + Send>;
    takes_dyn_trait_send(dyn_trait_send_send);

    // 3. Aliases to the trait are transparent.
    let dyn_trait_send = Box::new(Struct) as Box<dyn Trait + Send>;
    takes_dyn_trait_send_sendalias(dyn_trait_send);

    // 4. Calling an impl that duplicates an auto trait.
    let dyn_trait_send = Box::new(Struct) as Box<dyn Trait + Send>;
    dyn_trait_send.do_nothing();
}
