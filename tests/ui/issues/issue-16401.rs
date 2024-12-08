struct Slice<T> {
    data: *const T,
    len: usize,
}

fn main() {
    match () {
        Slice { data: data, len: len } => (),
        //~^ ERROR mismatched types
        //~| expected unit type `()`
        //~| found struct `Slice<_>`
        //~| expected `()`, found `Slice<_>`
        _ => unreachable!()
    }
}
