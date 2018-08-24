fn main() {
    let f  = || -> isize {
        let i: isize;
        i //~ ERROR use of possibly uninitialized variable: `i`
    };
    println!("{}", f());
}
