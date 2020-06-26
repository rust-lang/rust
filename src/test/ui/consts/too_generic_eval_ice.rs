pub struct Foo<A, B>(A, B);

impl<A, B> Foo<A, B> {
    const HOST_SIZE: usize = std::mem::size_of::<B>();

    pub fn crash() -> bool {
        [5; Self::HOST_SIZE] == [6; 0]
        //~^ ERROR constant expression depends on a generic parameter
        //~| ERROR binary operation `==` cannot be applied to type `[{integer}; _]`
    }
}

fn main() {}
