#![crate_name = "time"]

fn main() {
    let items = Box::new(vec![]); //~ ERROR E0282
    //~^ NOTE type must be known at this point
    //~| NOTE this is an inference error on crate `time` caused by an API change in Rust 1.80.0; update `time` to version `>=0.3.35`
    items.into();
}
