// check-fail
// known-bug

// This should pass, but it requires `Sized` to be coinductive.

#![feature(generic_associated_types)]

trait Allocator {
    type Allocated<T>;
}

enum LinkedList<A: Allocator> {
    Head,
    Next(A::Allocated<Self>)
}

fn main() {}
