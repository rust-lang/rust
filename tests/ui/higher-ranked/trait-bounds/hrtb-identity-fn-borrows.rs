// Test that the `'a` in the where clause correctly links the region
// of the output to the region of the input.

trait FnLike<A,R> {
    fn call(&self, arg: A) -> R;
}

fn call_repeatedly<F>(f: F)
    where F : for<'a> FnLike<&'a isize, &'a isize>
{
    // Result is stored: cannot re-assign `x`
    let mut x = 3;
    let y = f.call(&x);
    x = 5; //~ ERROR cannot assign to `x` because it is borrowed

    // Result is not stored: can re-assign `x`
    let mut x = 3;
    f.call(&x);
    f.call(&x);
    f.call(&x);
    x = 5;
    drop(y);
}

fn main() {
}
