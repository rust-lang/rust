fn test() {
    let v: isize;
    //~^ HELP consider making this binding mutable
    //~| SUGGESTION mut v
    v = 1; //~ NOTE first assignment
    println!("v={}", v);
    v = 2; //~ ERROR cannot assign twice to immutable variable
           //~| NOTE cannot assign twice to immutable
    println!("v={}", v);
}

fn main() {
}
