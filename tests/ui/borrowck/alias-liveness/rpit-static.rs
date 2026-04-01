//@ check-pass

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

fn foo(x: &mut i32) -> impl Sized + Captures<'_> + 'static {}

fn overlapping_mut() {
    let i = &mut 1;
    let x = foo(i);
    let y = foo(i);
}

fn live_past_borrow() {
    let y;
    {
        let x = &mut 1;
        y = foo(x);
    }
}

fn main() {}
