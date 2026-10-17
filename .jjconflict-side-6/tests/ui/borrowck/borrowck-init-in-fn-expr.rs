fn main() {
    let f  = || -> isize {
        let i: isize;
        i //~ ERROR E0381
    };
    println!("{}", f());
}
