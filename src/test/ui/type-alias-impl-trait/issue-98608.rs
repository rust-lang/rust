fn hi() -> impl Sized { std::ptr::null::<u8>() }

fn main() {
    let b: Box<dyn Fn() -> Box<u8>> = Box::new(hi);
    //~^ ERROR type mismatch resolving `<fn() -> impl Sized {hi} as FnOnce<()>>::Output == Box<u8>`
    let boxed = b();
    let null = *boxed;
    println!("{null:?}");
}
