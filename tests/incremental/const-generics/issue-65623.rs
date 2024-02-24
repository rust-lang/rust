//@ revisions:rpass1
pub struct Foo<T, const N: usize>([T; 0]);

impl<T, const N: usize> Foo<T, {N}> {
    pub fn new() -> Self {
        Foo([])
    }
}

fn main() {
    let _: Foo<u32, 0> = Foo::new();
}
