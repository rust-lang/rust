  //! Checks that `#[rustc_trivial_field_reads]` applies per method
  //! (issue #160621)

  #![feature(rustc_attrs)]
  #![deny(dead_code)]

  trait Access {
      fn get_a(&self) -> u32;
      fn get_b(&self) -> u32;
  }

  struct S {
    a: u32, //~ ERROR field `a` is never read
    b: u32
  }

  impl Access for S {
      #[rustc_trivial_field_reads]
      fn get_a(&self) -> u32 {
          self.a
      }

      fn get_b(&self) -> u32 {
          self.b
      }
  }

  struct T {
    a: u32,
    b: u32
  }

  impl Access for T {
    fn get_a(&self) -> u32 {
        self.a
    }

    fn get_b(&self) -> u32 {
        self.b
    }
  }

  fn main() {
    let s = S {
        a: 0,
        b: 0
    };

    let _ = s.get_a();
    let _ = s.get_b();

    let t = T {
        a: 0,
        b: 0
    };

    let _ = t.get_a();
    let _ = t.get_b();
  }
