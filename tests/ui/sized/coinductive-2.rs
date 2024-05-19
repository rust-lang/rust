//@ run-pass
struct Node<C: CollectionFactory<Self>> {
    _children: C::Collection,
}

trait CollectionFactory<T> {
    type Collection;
}

impl<T> CollectionFactory<T> for Vec<()> {
    type Collection = Vec<T>;
}

trait Collection<T>: Sized { //~ WARN trait `Collection` is never used
    fn push(&mut self, v: T);
}

impl<T> Collection<T> for Vec<T> {
    fn push(&mut self, v: T) {
        self.push(v)
    }
}

fn main() {
    let _ = Node::<Vec<()>> {
        _children: Vec::new(),
    };
}
