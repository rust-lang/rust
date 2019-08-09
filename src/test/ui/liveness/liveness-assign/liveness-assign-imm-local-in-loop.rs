fn test() {
    let v: isize;
    //~^ HELP make this binding mutable
    //~| SUGGESTION mut v
    loop {
        v = 1; //~ ERROR cannot assign twice to immutable variable `v`
               //~| NOTE cannot assign twice to immutable variable
    }
}

fn main() {
}
