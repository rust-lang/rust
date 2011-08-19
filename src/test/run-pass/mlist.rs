// -*- rust -*-
tag mlist { cons(int, @mlist); nil; }

fn main() { cons(10, @cons(11, @cons(12, @nil))); }
