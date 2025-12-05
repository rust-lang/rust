//@ dont-require-annotations: NOTE

struct Slice<T> {
    data: *const T,
    len: usize,
}

fn main() {
    let Slice { data: data, len: len } = "foo";
    //~^ ERROR mismatched types
    //~| NOTE found struct `Slice<_>`
}
