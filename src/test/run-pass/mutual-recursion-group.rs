


// -*- rust -*-
tag colour { red; green; blue; }

tag tree { children(@list); leaf(colour); }

tag list { cons(@tree, @list); nil; }

tag small_list { kons(int, @small_list); neel; }

fn main() { }
