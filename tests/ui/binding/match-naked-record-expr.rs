//@ run-pass

struct X { x: isize }

pub fn main() {
    let _x = match 0 {
      _ => X {
        x: 0
      }.x
    };
}
