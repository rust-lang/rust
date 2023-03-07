fn test() {
    let b = Box::new(1); //~ NOTE first assignment
                         //~| HELP consider making this binding mutable
                         //~| SUGGESTION mut b
    drop(b);
    b = Box::new(2); //~ ERROR cannot assign twice to immutable variable `b`
                     //~| NOTE cannot assign twice to immutable
                     //~| NOTE in this expansion of desugaring of drop and replace
    drop(b);
}

fn main() {
}
