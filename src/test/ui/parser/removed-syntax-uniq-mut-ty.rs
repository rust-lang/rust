// compile-flags: -Z parse-only

type mut_box = Box<mut isize>; //~ ERROR expected one of `>`, lifetime, or type, found `mut`
