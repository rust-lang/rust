//@ run-pass

#![allow(non_camel_case_types)]
#![allow(dead_code)]


enum colour { red, green, blue, }

enum tree { children(Box<list>), leaf(colour), }

enum list { cons(Box<tree>, Box<list>), nil, }

enum small_list { kons(isize, Box<small_list>), neel, }

pub fn main() { }
