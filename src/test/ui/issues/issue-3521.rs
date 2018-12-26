fn main() {
    let foo = 100;

    #[derive(Debug)]
    enum Stuff {
        Bar = foo
        //~^ ERROR attempt to use a non-constant value in a constant
    }

    println!("{:?}", Stuff::Bar);
}
