struct Clam {
    x: Box<isize>,
    y: Box<isize>,
}



struct Fish {
    a: Box<isize>,
}

fn main() {
    let a: Clam = Clam{ x: Box::new(1), y: Box::new(2) };
    let b: Clam = Clam{ x: Box::new(10), y: Box::new(20) };
    let z: isize = a.x + b.y;
    //~^ ERROR cannot add `Box<isize>` to `Box<isize>`
    println!("{}", z);
    assert_eq!(z, 21);
    let forty: Fish = Fish{ a: Box::new(40) };
    let two: Fish = Fish{ a: Box::new(2) };
    let answer: isize = forty.a + two.a;
    //~^ ERROR cannot add `Box<isize>` to `Box<isize>`
    println!("{}", answer);
    assert_eq!(answer, 42);
}
