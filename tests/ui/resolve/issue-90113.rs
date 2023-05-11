mod list {
    pub use self::List::Cons;

    pub enum List<T> {
        Cons(T, Box<List<T>>),
    }
}

mod alias {
    use crate::list::List;

    pub type Foo = List<String>;
}

fn foo(l: crate::alias::Foo) {
    match l {
        Cons(..) => {} //~ ERROR: cannot find tuple struct or tuple variant `Cons` in this scope
    }
}

fn main() {}
