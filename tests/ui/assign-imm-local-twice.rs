//@ check-pass
fn test() {
    let v: isize;
    //~^ HELP consider making this binding mutable
    //~| SUGGESTION mut
    v = 1; //~ NOTE first assignment
    println!("v={}", v);
    v = 2; //~ WARNING cannot assign twice to immutable variable
           //~| NOTE cannot assign twice to immutable
           //~| NOTE on by default
    println!("v={}", v);
}

fn main() {
}
