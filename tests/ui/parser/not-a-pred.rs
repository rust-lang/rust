fn f(a: isize, b: isize) : lt(a, b) { }
//~^ ERROR return types are denoted using `->`
//~| ERROR expected type, found function `lt` [E0573]
//~| ERROR expected type, found local variable `a` [E0573]
//~| ERROR expected type, found local variable `b` [E0573]

fn lt(a: isize, b: isize) { }

fn main() {
    let a: isize = 10;
    let b: isize = 23;
    check (lt(a, b));
    //~^ ERROR cannot find function `check` in this scope [E0425]
    f(a, b);
}
