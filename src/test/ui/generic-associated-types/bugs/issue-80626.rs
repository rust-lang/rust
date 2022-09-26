// check-fail
// known-bug: #80626

// This should pass, but it requires `Sized` to be coinductive.

trait Allocator {
    type Allocated<T>;
}

enum LinkedList<A: Allocator> {
    Head,
    Next(A::Allocated<Self>)
}

fn main() {}
