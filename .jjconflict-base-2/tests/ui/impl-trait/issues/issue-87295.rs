trait Trait {
    type Output;
}
impl Trait for () {
    type Output = i32;
}

struct Struct<F>(F);
impl<F> Struct<F> {
    pub fn new(_: F) -> Self {
        todo!()
    }
}

fn main() {
    let _do_not_waste: Struct<impl Trait<Output = i32>> = Struct::new(());
    //~^ ERROR `impl Trait` is not allowed in the type of variable bindings
}
