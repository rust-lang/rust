//@ run-pass

struct Foo<const N: usize = 2>;
impl<const N: usize> Foo<N> {
    fn make_arr() -> [(); N] {
        [(); N]
    }
}

fn main() {
    let [(), (), ()] = Foo::<_>::make_arr();
}
