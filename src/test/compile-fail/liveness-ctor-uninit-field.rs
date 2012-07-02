class cat {
  let mut a: int;
  let mut b: int;
  let mut c: int;

  new() {
     self.a = 3;
     self.b = self.a;
     self.a += self.c; //~ ERROR use of possibly uninitialized field: `self.c`
  }
}

fn main() {
}
