fn test() {
    let v: isize;
    //~^ HELP consider making this binding mutable
    //~| SUGGESTION mut
    v = 2;  //~ NOTE first assignment
    v += 1; //~ ERROR cannot assign twice to immutable variable `v`
            //~| NOTE cannot assign twice to immutable
    v.clone();
}

fn main() {
}
