fn test() {
    let v: isize = 1; //~ NOTE first assignment
                      //~| HELP make this binding mutable
                      //~| SUGGESTION mut v
    v.clone();
    v = 2; //~ ERROR cannot assign twice to immutable variable `v`
           //~| NOTE cannot assign twice to immutable
    v.clone();
}

fn main() {
}
