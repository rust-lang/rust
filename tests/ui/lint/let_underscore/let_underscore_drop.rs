//@ check-pass
#![warn(let_underscore_drop)]

struct NontrivialDrop;

impl Drop for NontrivialDrop {
    fn drop(&mut self) {
        println!("Dropping!");
    }
}

fn identity<T>(t: T) -> T {
    t
}

fn main() {
    let _ = NontrivialDrop; //~WARNING non-binding let on a type that has a destructor

    let _ = *&NontrivialDrop; //~WARNING non-binding let on a type that has a destructor

    let _ = *identity(&NontrivialDrop); //~WARNING non-binding let on a type that has a destructor

    // FIXME: nothing is dropped here, but the lint can't see through the call.
    let d = NontrivialDrop;
    let _ = *identity(&d); //~WARNING non-binding let on a type that has a destructor

    let (_, _) = (NontrivialDrop, NontrivialDrop); // This should be ignored.

    let boxed = Box::new(NontrivialDrop);
    let _: Box<dyn Send> = boxed; //~WARNING non-binding let on a type that has a destructor
}
