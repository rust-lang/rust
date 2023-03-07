// run-pass

struct F { f: Vec<isize> }

fn impure(_v: &[isize]) {
}

pub fn main() {
    let mut x = F {f: vec![3]};

    match x {
      F {f: ref mut v} => {
        impure(v);
      }
    }
}
