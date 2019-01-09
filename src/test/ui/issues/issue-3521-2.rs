fn main() {
    let foo = 100;

    static y: isize = foo + 1;
    //~^ ERROR can't capture dynamic environment

    println!("{}", y);
}
