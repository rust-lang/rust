// check-pass

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

fn foo(x: &i32) -> impl Sized + Captures<'_> + 'static {}

fn main() {
    let y;
    {
        let x = 1;
        y = foo(&x);
    }
}
