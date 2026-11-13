//@ check-pass

trait Foo {
    fn rpitit(&mut self) -> impl Sized + 'static;
}

fn live_past_borrow<T: Foo>(mut t: T) {
    let x = t.rpitit();
    drop(t);
    drop(x);
}

fn overlapping_mut<T: Foo>(mut t: T) {
    let a = t.rpitit();
    let b = t.rpitit();
}

fn main() {}
