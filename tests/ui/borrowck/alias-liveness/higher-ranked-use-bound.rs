//@ check-pass

trait Outlives<'a>: 'a {}
impl<'a, T: 'a> Outlives<'a> for T {}

fn test<'o>(v: &'o Vec<i32>) -> impl use<'o> + for<'a> Outlives<'a> {}

fn opaque_doesnt_use_temporary() {
    let a = test(&vec![]);
}

fn main() {}
