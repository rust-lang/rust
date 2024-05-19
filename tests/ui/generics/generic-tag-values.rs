//@ run-pass
#![allow(non_camel_case_types)]

enum noption<T> { some(T), }

struct Pair { x: isize, y: isize }

pub fn main() {
    let nop: noption<isize> = noption::some::<isize>(5);
    match nop { noption::some::<isize>(n) => { println!("{}", n); assert_eq!(n, 5); } }
    let nop2: noption<Pair> = noption::some(Pair{x: 17, y: 42});
    match nop2 {
      noption::some(t) => {
        println!("{}", t.x);
        println!("{}", t.y);
        assert_eq!(t.x, 17);
        assert_eq!(t.y, 42);
      }
    }
}
