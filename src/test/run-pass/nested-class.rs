fn main() {
  
  class b {
    let i: int;
    fn do_stuff() -> int { return 37; }
    new(i:int) { self.i = i; }
  }

  //  fn b(x:int) -> int { fail; }

  let z = b(42);
  assert(z.i == 42);
  assert(z.do_stuff() == 37);
  
}