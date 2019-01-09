// Regression test for issue #45045

#![feature(nll)]

enum Xyz {
    A,
    B,
}

fn main() {
    let mut e = Xyz::A;
    let f = &mut e;
    let g = f;
    match e {
        Xyz::A => println!("a"),
        //~^ cannot use `e` because it was mutably borrowed [E0503]
        Xyz::B => println!("b"),
    };
    *g = Xyz::B;
}
