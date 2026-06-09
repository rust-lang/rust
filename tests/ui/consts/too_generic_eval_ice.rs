//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

pub struct Foo<A, B>(A, B);

impl<A, B> Foo<A, B> {
    const HOST_SIZE: usize = std::mem::size_of::<B>();

    pub fn crash() -> bool {
        [5; Self::HOST_SIZE] == [6; 0]
        //[current]~^ ERROR constant expression depends on a generic parameter
        //[current]~| ERROR constant expression depends on a generic parameter
        //[current]~| ERROR constant expression depends on a generic parameter
        //[current]~| ERROR can't compare `[{integer}; Self::HOST_SIZE]` with `[{integer}; 0]`
        //[next]~^^^^^ ERROR type annotations needed
    }
}

fn main() {}
