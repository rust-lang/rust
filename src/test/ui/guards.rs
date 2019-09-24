// run-pass

#![allow(non_shorthand_field_patterns)]

#[derive(Copy, Clone)]
struct Pair { x: isize, y: isize }

pub fn main() {
    let a: isize =
        match 10 { x if x < 7 => { 1 } x if x < 11 => { 2 } 10 => { 3 } _ => { 4 } };
    assert_eq!(a, 2);

    let b: isize =
        match (Pair {x: 10, y: 20}) {
          x if x.x < 5 && x.y < 5 => { 1 }
          Pair {x: x, y: y} if x == 10 && y == 20 => { 2 }
          Pair {x: _x, y: _y} => { 3 }
        };
    assert_eq!(b, 2);
}
