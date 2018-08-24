// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

fn f(y: Box<isize>) {
    *y = 5; //[ast]~ ERROR cannot assign
            //[mir]~^ ERROR cannot assign
}

fn g() {
    let _frob = |q: Box<isize>| { *q = 2; }; //[ast]~ ERROR cannot assign
    //[mir]~^ ERROR cannot assign
}

fn main() {}
