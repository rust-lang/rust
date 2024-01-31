// edition:2021

#![feature(async_closure)]

fn main() {
    let x = async move |x: &str| {
        //~^ ERROR lifetime may not live long enough
        // This error is proof that the `&str` type is higher-ranked.
        // This won't work until async closures are fully impl'd.
        println!("{x}");
    };
}
