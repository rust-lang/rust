fn id<T>(x: Box<T>) -> Box<T> { x }

fn main() {
    let mut m = 0;
    let b = unsafe { Box::from_raw(&mut m) };
    let mut b2 = id(b);
    // Since `id` returns a `Box`, this should invalidate all other pointers to this memory.
    *b2 = 5;
    std::mem::forget(b2);
    println!("{}", m); //~ERROR: tag does not exist in the borrow stack
}
