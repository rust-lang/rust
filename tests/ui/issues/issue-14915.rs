fn main() {
    let x: Box<isize> = Box::new(0);

    println!("{}", x + 1);
    //~^ ERROR cannot add `{integer}` to `Box<isize>`
}
