fn f(y: Box<isize>) {
    *y = 5; //~ ERROR cannot assign
}

fn g() {
    let _frob = |q: Box<isize>| { *q = 2; }; //~ ERROR cannot assign
}

fn main() {}
