// revisions: ast mir
//[mir]compile-flags: -Zborrowck=mir

fn test() {
    let b = Box::new(1); //[ast]~ NOTE first assignment
                         //[mir]~^ NOTE first assignment
                         //[mir]~| HELP make this binding mutable
                         //[mir]~| SUGGESTION mut b
    drop(b);
    b = Box::new(2); //[ast]~ ERROR cannot assign twice to immutable variable
                     //[mir]~^ ERROR cannot assign twice to immutable variable `b`
                     //[ast]~| NOTE cannot assign twice to immutable
                     //[mir]~| NOTE cannot assign twice to immutable
    drop(b);
}

fn main() {
}
