//@ check-pass
fn test() {
    let v: isize = 1; //~ NOTE first assignment
                      //~| HELP consider making this binding mutable
                      //~| SUGGESTION mut
    v.clone();
    v = 2; //~ WARNING cannot assign twice to immutable variable `v`
           //~| NOTE cannot assign twice to immutable
           //~| NOTE on by default
    v.clone();
}

fn main() {
}
