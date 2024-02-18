//@ check-pass

fn make<T>() -> T {
    panic!()
}

fn take<T>(x: T) {}

fn main() {
    let x: for<'a> fn(&'a u32) -> _ = make();
    let y: &'static u32 = x(&22);
    take::<for<'b> fn(&'b u32) -> &'b u32>(x);
}
