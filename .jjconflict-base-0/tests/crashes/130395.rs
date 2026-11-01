//@ known-bug: #130395
//@ needs-rustc-debug-assertions

enum U {
    B(isize, usize),
}

fn main() {
    let x = T::A(U::C);
}
