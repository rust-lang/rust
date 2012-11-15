// -*- rust -*-

// error-pattern: illegal recursive enum type; wrap the inner value in a box

enum mlist { cons(int, mlist), nil, }

fn main() { let a = cons(10, cons(11, nil)); }