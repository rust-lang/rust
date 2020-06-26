enum MList { Cons(isize, MList), Nil }
//~^ ERROR recursive type `MList` has infinite size
//~| ERROR cycle detected when computing drop-check constraints for `MList`

fn main() { let a = MList::Cons(10, MList::Cons(11, MList::Nil)); }
