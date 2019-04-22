fn test() {
    let v: isize;
    //~^ HELP make this binding mutable
    //~| SUGGESTION mut v
    v = 2;  //~ NOTE first assignment
    v += 1; //~ ERROR cannot assign twice to immutable variable `v`
            //~| NOTE cannot assign twice to immutable
    v.clone();
}

fn main() {
}
