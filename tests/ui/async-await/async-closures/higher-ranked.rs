// edition:2021
// check-pass

#![feature(async_closure)]

fn main() {
    let x = async move |x: &str| {
        println!("{x}");
    };
}
