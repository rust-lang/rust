//@ check-pass

trait Allocator {
    type Allocated<T>;
}

enum LinkedList<A: Allocator> {
    Head,
    Next(A::Allocated<Self>),
}

fn main() {}
