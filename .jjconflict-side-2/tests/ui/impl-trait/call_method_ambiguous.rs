//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ run-pass

trait Get {
    fn get(&mut self) -> u32;
}

impl Get for () {
    fn get(&mut self) -> u32 {
        0
    }
}

impl<T> Get for &mut T
where
    T: Get,
{
    fn get(&mut self) -> u32 {
        T::get(self) + 1
    }
}

fn foo(n: usize, m: &mut ()) -> impl Get + use<'_> {
    if n > 0 {
        let mut iter = foo(n - 1, m);
        assert_eq!(iter.get(), 1);
    }
    m
}

fn main() {
    let g = foo(1, &mut ()).get();
    assert_eq!(g, 1);
}
