#![feature(explicit_tail_calls)]
#![expect(incomplete_features)]

fn link(x: &str) -> &'static str {
    become passthrough(x);
    //~^ ERROR lifetime may not live long enough
}

fn passthrough<T>(t: T) -> T { t }

fn main() {
    let x = String::from("hello, world");
    let s = link(&x);
    drop(x);
    println!("{s}");
}
