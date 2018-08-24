#![feature(nll)]

fn main() {
    let _vec: Vec<&'static String> = vec![&String::new()];
    //~^ ERROR borrowed value does not live long enough [E0597]
}
