// run-rustfix
fn main() {
    #[allow(non_upper_case_globals)]
    let foo: isize = 100;

    #[derive(Debug)]
    enum Stuff {
        Bar = foo
        //~^ ERROR attempt to use a non-constant value in a constant
        //~| ERROR failed to evaluate
    }

    println!("{:?}", Stuff::Bar);
}
