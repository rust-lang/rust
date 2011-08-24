// -*- rust -*-

tag list { cons(int, @list); nil; }

type bubu = {x: int, y: int};

pure fn less_than(x: int, y: int) -> bool { ret x < y; }

type ordered_range = {low: int, high: int}  : less_than(*.low, *.high);

fn main() { }
