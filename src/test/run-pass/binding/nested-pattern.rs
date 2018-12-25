// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

// a bug was causing this to complain about leaked memory on exit

enum t { foo(isize, usize), bar(isize, Option<isize>), }

fn nested(o: t) {
    match o {
        t::bar(_i, Some::<isize>(_)) => { println!("wrong pattern matched"); panic!(); }
        _ => { println!("succeeded"); }
    }
}

pub fn main() { nested(t::bar(1, None::<isize>)); }
