// run-pass
// A basic test of using a higher-ranked trait bound.


trait FnLike<A,R> {
    fn call(&self, arg: A) -> R;
}

struct Identity;

impl<'a, T> FnLike<&'a T, &'a T> for Identity {
    fn call(&self, arg: &'a T) -> &'a T {
        arg
    }
}

fn call_repeatedly<F>(f: F)
    where F : for<'a> FnLike<&'a isize, &'a isize>
{
    let x = 3;
    let y = f.call(&x);
    assert_eq!(3, *y);
}

fn main() {
    call_repeatedly(Identity);
}
