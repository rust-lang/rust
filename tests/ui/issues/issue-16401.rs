struct Slice<T> {
    data: *const T,
    len: usize,
}

fn main() {
    match () {
        Slice { data: data, len: len } => (),
        //~^ ERROR mismatched types
        //~| NOTE_NONVIRAL expected unit type `()`
        //~| NOTE_NONVIRAL found struct `Slice<_>`
        //~| NOTE_NONVIRAL expected `()`, found `Slice<_>`
        _ => unreachable!()
    }
}
