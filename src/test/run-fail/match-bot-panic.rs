// error-pattern:explicit panic

#![allow(unreachable_code)]
#![allow(unused_variables)]

fn foo(s: String) {}

fn main() {
    let i = match Some::<isize>(3) {
        None::<isize> => panic!(),
        Some::<isize>(_) => panic!(),
    };
    foo(i);
}
