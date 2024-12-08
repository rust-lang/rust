fn main() {}
//@no-rustfix
fn no_panic<T>(slice: &[T]) {
    let mut iter = slice.iter();
    loop {
        //~^ ERROR: this loop never actually loops
        //~| ERROR: this loop could be written as a `while let` loop
        //~| NOTE: `-D clippy::while-let-loop` implied by `-D warnings`
        let _ = match iter.next() {
            Some(ele) => ele,
            None => break,
        };
        loop {}
        //~^ ERROR: empty `loop {}` wastes CPU cycles
    }
}
