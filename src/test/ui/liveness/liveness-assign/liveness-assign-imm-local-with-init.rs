// revisions: ast mir
//[mir]compile-flags: -Zborrowck=mir

fn test() {
    let v: isize = 1; //[ast]~ NOTE first assignment
                      //[mir]~^ NOTE first assignment
                      //[mir]~| HELP make this binding mutable
                      //[mir]~| SUGGESTION mut v
    v.clone();
    v = 2; //[ast]~ ERROR cannot assign twice to immutable variable
           //[mir]~^ ERROR cannot assign twice to immutable variable `v`
           //[ast]~| NOTE cannot assign twice to immutable
           //[mir]~| NOTE cannot assign twice to immutable
    v.clone();
}

fn main() {
}
