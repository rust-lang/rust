//! Test that mutually recursive type definitions are properly handled by the compiler.
//! This checks that types can reference each other in their definitions through
//! `Box` indirection, creating cycles in the type dependency graph.

//@ run-pass

#[derive(Debug, PartialEq)]
enum Colour {
    Red,
    Green,
    Blue,
}

#[derive(Debug, PartialEq)]
enum Tree {
    Children(Box<List>),
    Leaf(Colour),
}

#[derive(Debug, PartialEq)]
enum List {
    Cons(Box<Tree>, Box<List>),
    Nil,
}

#[derive(Debug, PartialEq)]
enum SmallList {
    Kons(isize, Box<SmallList>),
    Neel,
}

pub fn main() {
    // Construct and test all variants of Colour
    let _ = Tree::Leaf(Colour::Red);

    let _ = Tree::Leaf(Colour::Green);

    let _ = Tree::Leaf(Colour::Blue);

    let _ = List::Nil;

    let _ = Tree::Children(Box::new(List::Nil));

    let _ = List::Cons(Box::new(Tree::Leaf(Colour::Blue)), Box::new(List::Nil));

    let _ = SmallList::Kons(42, Box::new(SmallList::Neel));
}
