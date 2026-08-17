trait Captures<'a> {}
impl<T> Captures<'_> for T {}

fn foo(x: &mut i32) -> impl Sized + Captures<'_> + 'static {}

fn overlapping_mut() {
    let i = &mut 1;
    let x = foo(i);
    let y = foo(i); //~ ERROR cannot borrow `*i` as mutable more than once at a time
}

fn live_past_borrow() {
    let y;
    {
        let x = &mut 1; //~ ERROR temporary value dropped while borrowed
        y = foo(x);
    }
}

fn main() {}
