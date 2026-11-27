//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// <https://github.com/rust-lang/rust/issues/114936>
fn whoops(
    s: String,
    f: impl for<'s> FnOnce(&'s str) -> (&'static str, [&'static &'s (); 0]),
) -> &'static str
{
    f(&s).0
    //~^ ERROR `s` does not live long enough
}

// <https://github.com/rust-lang/rust/issues/118876>
fn extend<T>(input: &T) -> &'static T {
    struct Bounded<'a, 'b: 'static, T>(&'a T, [&'b (); 0]);
    let n: Box<dyn FnOnce(&T) -> Bounded<'static, '_, T>> = Box::new(|x| Bounded(x, []));
    n(input).0
    //~^ ERROR borrowed data escapes outside of function
}

// <https://github.com/rust-lang/rust/issues/118876>
fn extend_mut<'a, T>(input: &'a mut T) -> &'static mut T {
    struct Bounded<'a, 'b: 'static, T>(&'a mut T, [&'b (); 0]);
    let mut n: Box<dyn FnMut(&mut T) -> Bounded<'static, '_, T>> = Box::new(|x| Bounded(x, []));
    n(input).0
    //~^ ERROR borrowed data escapes outside of function
}

fn main() {}
