//@ check-pass
struct Node<C: Trait>(C::Assoc::<Self>);

trait Trait {
    type Assoc<T>;
}

impl Trait for Vec<()> {
    type Assoc<T> = Vec<T>;
}

fn main() {
    let _ = Node::<Vec<()>>(Vec::new());
}
