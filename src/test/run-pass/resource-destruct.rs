class shrinky_pointer {
  let i: @@mut int;
  fn look_at() -> int { ret **(self.i); }
  new(i: @@mut int) { self.i = i; }
  drop { log(error, ~"Hello!"); **(self.i) -= 1; }
}

fn main() {
    let my_total = @@mut 10;
    { let pt <- shrinky_pointer(my_total); assert (pt.look_at() == 10); }
    log(error, #fmt("my_total = %d", **my_total));
    assert (**my_total == 9);
}
