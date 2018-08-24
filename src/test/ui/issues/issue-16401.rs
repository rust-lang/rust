struct Slice<T> {
    data: *const T,
    len: usize,
}

fn main() {
    match () {
        Slice { data: data, len: len } => (),
        //~^ ERROR mismatched types
        //~| expected type `()`
        //~| found type `Slice<_>`
        //~| expected (), found struct `Slice`
        _ => unreachable!()
    }
}
