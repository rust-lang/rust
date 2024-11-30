//@ check-pass
fn test() {
    let b = Box::new(1); //~ NOTE first assignment
                         //~| HELP consider making this binding mutable
                         //~| SUGGESTION mut
    drop(b);
    b = Box::new(2); //~ WARNING cannot assign twice to immutable variable `b`
                     //~| NOTE cannot assign twice to immutable
                     //~| NOTE on by default
    drop(b);
}

fn main() {
}
