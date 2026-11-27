fn test() {
    let v: isize;
    //~^ HELP consider making this binding mutable
    //~| SUGGESTION mut
    loop {
        v = 1; //~ ERROR cannot assign twice to immutable variable `v`
               //~| NOTE cannot assign twice to immutable variable
    }
}

fn main() {
}
