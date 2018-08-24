#![feature(box_syntax)]

struct clam {
    x: Box<isize>,
    y: Box<isize>,
}

struct fish {
    a: Box<isize>,
}

fn main() {
    let a: clam = clam{x: box 1, y: box 2};
    let b: clam = clam{x: box 10, y: box 20};
    let z: isize = a.x + b.y;
    //~^ ERROR binary operation `+` cannot be applied to type `std::boxed::Box<isize>`
    println!("{}", z);
    assert_eq!(z, 21);
    let forty: fish = fish{a: box 40};
    let two: fish = fish{a: box 2};
    let answer: isize = forty.a + two.a;
    //~^ ERROR binary operation `+` cannot be applied to type `std::boxed::Box<isize>`
    println!("{}", answer);
    assert_eq!(answer, 42);
}
