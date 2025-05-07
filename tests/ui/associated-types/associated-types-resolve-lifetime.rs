//@ check-pass

trait Get<T> {
    fn get(&self) -> T;
}

trait Trait<'a> {
    type T: 'static;
    type U: Get<&'a isize>;

    fn dummy(&'a self) { }
}

fn main() {}
