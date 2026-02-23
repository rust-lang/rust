enum MList {
    //~^ ERROR recursive type `MList` has infinite size
    Cons(isize, MList),
    Nil,
}

fn main() {
    let a = MList::Cons(10, MList::Cons(11, MList::Nil));
}
