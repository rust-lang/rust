fn main() {
    let j = || -> isize {
        let i: isize;
        i //~ ERROR use of possibly uninitialized variable: `i`
    };
    j();
}
