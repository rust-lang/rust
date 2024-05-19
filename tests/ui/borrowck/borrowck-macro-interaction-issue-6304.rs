//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unconditional_recursion)]

// Check that we do not ICE when compiling this
// macro, which reuses the expression `$id`

#![feature(box_patterns)]

struct Foo {
  a: isize
}

pub enum Bar {
  Bar1, Bar2(isize, Box<Bar>),
}

impl Foo {
  fn elaborate_stm(&mut self, s: Box<Bar>) -> Box<Bar> {
    macro_rules! declare {
      ($id:expr, $rest:expr) => ({
        self.check_id($id);
        Box::new(Bar::Bar2($id, $rest))
      })
    }
    match s {
      box Bar::Bar2(id, rest) => declare!(id, self.elaborate_stm(rest)),
      _ => panic!()
    }
  }

  fn check_id(&mut self, s: isize) { panic!() }
}

pub fn main() { }
