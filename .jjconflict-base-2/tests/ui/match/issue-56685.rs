#![allow(dead_code)]
#![deny(unused_variables)]

// This test aims to check that unused variable suggestions update bindings in all
// match arms.

fn main() {
    enum E {
        A(i32,),
        B(i32,),
    }

    match E::A(1) {
        E::A(x) | E::B(x) => {}
        //~^ ERROR unused variable: `x`
    }

    enum F {
        A(i32, i32,),
        B(i32, i32,),
        C(i32, i32,),
    }

    let _ = match F::A(1, 2) {
        F::A(x, y) | F::B(x, y) => { y },
        //~^ ERROR unused variable: `x`
        F::C(a, b) => { 3 }
        //~^ ERROR unused variable: `a`
        //~^^ ERROR unused variable: `b`
    };

    let _ = if let F::A(x, y) | F::B(x, y) = F::A(1, 2) {
    //~^ ERROR unused variable: `x`
        y
    } else {
        3
    };

    while let F::A(x, y) | F::B(x, y) = F::A(1, 2) {
    //~^ ERROR unused variable: `x`
        let _ = y;
        break;
    }
}
