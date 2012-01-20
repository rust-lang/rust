


// -*- rust -*-
enum colour { red; green; blue; }

enum tree { children(@list); leaf(colour); }

enum list { cons(@tree, @list); nil; }

enum small_list { kons(int, @small_list); neel; }

fn main() { }
