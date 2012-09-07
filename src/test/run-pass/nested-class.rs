fn main() {
  
  struct b {
    i: int,
    fn do_stuff() -> int { return 37; }
  }

    fn b(i:int) -> b {
        b {
            i: i
        }
    }

  //  fn b(x:int) -> int { fail; }

  let z = b(42);
  assert(z.i == 42);
  assert(z.do_stuff() == 37);
  
}