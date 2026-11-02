pub trait Tr {}
impl Tr for u32 {}

pub fn foo() -> Box<impl Tr + ?Sized> {
    Box::new(1u32)
}

fn main() {
    let _: Box<dyn Tr> = foo();
    //~^ ERROR: the size for values of type `impl Tr + ?Sized` cannot be known

}
