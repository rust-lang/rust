fn main() {}

fn no_panic<T>(slice: &[T]) {
    let mut iter = slice.iter();
    loop {
        let _ = match iter.next() {
            Some(ele) => ele,
            None => break,
        };
        loop {}
    }
}
