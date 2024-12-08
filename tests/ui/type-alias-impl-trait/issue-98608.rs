fn hi() -> impl Sized {
    std::ptr::null::<u8>()
}

fn main() {
    let b: Box<dyn Fn() -> Box<u8>> = Box::new(hi);
    //~^ ERROR expected `hi` to be a fn item that returns `Box<u8>`, but it returns `impl Sized`
    let boxed = b();
    let null = *boxed;
    println!("{null:?}");
}
