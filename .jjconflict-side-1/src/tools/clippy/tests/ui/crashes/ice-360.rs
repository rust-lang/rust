fn main() {}
//@no-rustfix
fn no_panic<T>(slice: &[T]) {
    let mut iter = slice.iter();
    loop {
        //~^ never_loop
        //~| while_let_loop

        let _ = match iter.next() {
            Some(ele) => ele,
            None => break,
        };
        loop {}
        //~^ empty_loop
    }
}
