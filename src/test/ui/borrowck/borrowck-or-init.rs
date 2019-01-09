fn main() {
    let i: isize;

    println!("{}", false || { i = 5; true });
    println!("{}", i); //~ ERROR use of possibly uninitialized variable: `i`
}
