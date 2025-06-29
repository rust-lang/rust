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
    let leaf_red = Tree::Leaf(Colour::Red);
    assert_eq!(leaf_red, Tree::Leaf(Colour::Red));

    let leaf_green = Tree::Leaf(Colour::Green);
    assert_eq!(leaf_green, Tree::Leaf(Colour::Green));

    let leaf_blue = Tree::Leaf(Colour::Blue);
    assert_eq!(leaf_blue, Tree::Leaf(Colour::Blue));

    let empty_list = List::Nil;
    assert_eq!(empty_list, List::Nil);

    let tree_with_children = Tree::Children(Box::new(List::Nil));
    assert_eq!(tree_with_children, Tree::Children(Box::new(List::Nil)));

    let list_with_tree = List::Cons(Box::new(Tree::Leaf(Colour::Blue)), Box::new(List::Nil));
    assert_eq!(list_with_tree, List::Cons(Box::new(Tree::Leaf(Colour::Blue)), Box::new(List::Nil)));

    let small_list = SmallList::Kons(42, Box::new(SmallList::Neel));
    assert_eq!(small_list, SmallList::Kons(42, Box::new(SmallList::Neel)));
}
