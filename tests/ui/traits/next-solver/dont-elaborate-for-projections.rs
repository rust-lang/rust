//@ compile-flags: -Znext-solver
//@ check-pass

trait Iter<'a, I: 'a>: Iterator<Item = &'a I> {}

fn needs_iter<'a, T: Iter<'a, I> + ?Sized, I: 'a>(_: &T) {}

fn test(x: &dyn Iter<'_, ()>) {
    needs_iter(x);
}

fn main() {}
