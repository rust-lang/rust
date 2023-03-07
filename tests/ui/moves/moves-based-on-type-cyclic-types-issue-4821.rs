// Test for a subtle failure computing kinds of cyclic types, in which
// temporary kinds wound up being stored in a cache and used later.
// See rustc::ty::type_contents() for more information.


struct List { key: isize, next: Option<Box<List>> }

fn foo(node: Box<List>) -> isize {
    let r = match node.next {
        Some(right) => consume(right),
        None => 0
    };
    consume(node) + r //~ ERROR use of partially moved value: `node`
}

fn consume(v: Box<List>) -> isize {
    v.key
}

fn main() {}
