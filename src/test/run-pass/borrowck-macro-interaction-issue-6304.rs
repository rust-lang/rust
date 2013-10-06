// Check that we do not ICE when compiling this
// macro, which reuses the expression `$id`

#[feature(macro_rules)];

struct Foo {
  a: int
}

pub enum Bar {
  Bar1, Bar2(int, ~Bar),
}

impl Foo {
  fn elaborate_stm(&mut self, s: ~Bar) -> ~Bar {
    macro_rules! declare(
      ($id:expr, $rest:expr) => ({
        self.check_id($id);
        ~Bar2($id, $rest)
      })
    );
    match s {
      ~Bar2(id, rest) => declare!(id, self.elaborate_stm(rest)),
      _ => fail2!()
    }
  }

  fn check_id(&mut self, s: int) { fail2!() }
}

pub fn main() { }
