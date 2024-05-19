fn main() {
    let j = || -> isize {
        let i: isize;
        i //~ ERROR E0381
    };
    j();
}
