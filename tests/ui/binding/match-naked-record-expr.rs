// run-pass
// pretty-expanded FIXME #23616

struct X { x: isize }

pub fn main() {
    let _x = match 0 {
      _ => X {
        x: 0
      }.x
    };
}
