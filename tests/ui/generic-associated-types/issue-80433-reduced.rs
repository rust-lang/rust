//@ check-pass

struct E {}

trait TestMut {
    type Output<'a>;
    fn test_mut(&mut self) -> Self::Output<'static>;
}

impl TestMut for E {
    type Output<'a> = usize;
    fn test_mut(&mut self) -> Self::Output<'static> {
        todo!()
    }
}

fn test_simpler<'a>(_: impl TestMut<Output<'a> = usize>) {}

fn main() {
    test_simpler(E {});
}
