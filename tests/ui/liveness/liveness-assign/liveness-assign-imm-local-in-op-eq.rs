//@ check-pass
fn test() {
    let v: isize;
    //~^ HELP consider making this binding mutable
    //~| SUGGESTION mut
    v = 2;  //~ NOTE first assignment
    v += 1; //~ WARNING cannot assign twice to immutable variable `v`
            //~| NOTE cannot assign twice to immutable
            //~| NOTE on by default
    v.clone();
}

fn main() {
}
