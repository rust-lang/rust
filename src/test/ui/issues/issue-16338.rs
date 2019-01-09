struct Slice<T> {
    data: *const T,
    len: usize,
}

fn main() {
    let Slice { data: data, len: len } = "foo";
    //~^ ERROR mismatched types
    //~| found type `Slice<_>`
}
