//@ edition:2021
//@ check-pass

pub struct Example<'a, T> {
  a: T,
  b: &'a T,
}

impl<'a, T> Example<'a, T> {
  pub fn error_trying_to_destructure_self_in_closure(self) {
    let closure = || {
      let Self { a, b } = self;
    };
  }
}

fn main() {}
