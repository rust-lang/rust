fn f(a: isize, b: isize) : lt(a, b) { }
//~^ ERROR return types are denoted using `->`
//~| ERROR cannot find type `lt` in this scope [E0573]
//~| ERROR cannot find type `a` in this scope [E0573]
//~| ERROR cannot find type `b` in this scope [E0573]

fn lt(a: isize, b: isize) { }

fn main() {
    let a: isize = 10;
    let b: isize = 23;
    check (lt(a, b));
    //~^ ERROR cannot find function `check` in this scope [E0425]
    f(a, b);
}
