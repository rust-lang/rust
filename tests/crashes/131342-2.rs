//@ known-bug: #131342
// see also: 131342.rs

fn main() {
    problem_thingy(Once);
}

struct Once;

impl Iterator for Once {
    type Item = ();
}

fn problem_thingy(items: impl Iterator) {
    let peeker = items.peekable();
    problem_thingy(&peeker);
}

trait Iterator {
    type Item;

    fn peekable(self) -> Peekable<Self>
    where
        Self: Sized,
    {
        loop {}
    }
}

struct Peekable<I: Iterator> {
    _peeked: I::Item,
}

impl<I: Iterator> Iterator for Peekable<I> {
    type Item = I::Item;
}

impl<I: Iterator + ?Sized> Iterator for &I {
    type Item = I::Item;
}
