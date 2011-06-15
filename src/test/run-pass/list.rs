


// -*- rust -*-
tag list { cons(int, @list); nil; }

fn main() { cons(10, @cons(11, @cons(12, @nil))); }