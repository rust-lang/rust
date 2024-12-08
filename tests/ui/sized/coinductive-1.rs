//@ check-pass
struct Node<C: Trait<Self>>(C::Assoc);

trait Trait<T> {
    type Assoc;
}

impl<T> Trait<T> for Vec<()> {
    type Assoc = Vec<T>;
}

fn main() {
    let _ = Node::<Vec<()>>(Vec::new());
}
