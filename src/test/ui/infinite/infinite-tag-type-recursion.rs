enum MList { Cons(isize, MList), Nil }
//~^ ERROR recursive type `MList` has infinite size

fn main() { let a = MList::Cons(10, MList::Cons(11, MList::Nil)); }
