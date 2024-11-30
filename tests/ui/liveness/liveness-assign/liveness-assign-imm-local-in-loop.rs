//@ check-pass
fn test() {
    let v: isize;
    //~^ HELP consider making this binding mutable
    //~| SUGGESTION mut
    loop {
        v = 1; //~ WARNING cannot assign twice to immutable variable `v`
               //~| NOTE cannot assign twice to immutable variable
               //~| NOTE on by default
    }
}

fn main() {
}
