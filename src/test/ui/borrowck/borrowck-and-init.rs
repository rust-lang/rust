fn main() {
    let i: isize;

    println!("{}", false && { i = 5; true });
    println!("{}", i); //~ ERROR borrow of possibly uninitialized variable: `i`
}
