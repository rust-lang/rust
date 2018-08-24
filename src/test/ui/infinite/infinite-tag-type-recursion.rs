enum mlist { cons(isize, mlist), nil, }
//~^ ERROR recursive type `mlist` has infinite size

fn main() { let a = mlist::cons(10, mlist::cons(11, mlist::nil)); }
