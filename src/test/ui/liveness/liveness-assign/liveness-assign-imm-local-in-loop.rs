// revisions: ast mir
//[mir]compile-flags: -Zborrowck=mir

fn test() {
    let v: isize;
    //[mir]~^ NOTE consider changing this to `mut v`
    loop {
        v = 1; //[ast]~ ERROR cannot assign twice to immutable variable
               //[mir]~^ ERROR cannot assign twice to immutable variable `v`
               //[ast]~| NOTE cannot assign twice to immutable variable
               //[mir]~| NOTE cannot assign twice to immutable variable
        v.clone(); // just to prevent liveness warnings
    }
}

fn main() {
}
