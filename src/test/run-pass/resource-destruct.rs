struct shrinky_pointer {
  i: @@mut int,
  drop { log(error, ~"Hello!"); **(self.i) -= 1; }
}

impl shrinky_pointer {
  fn look_at() -> int { return **(self.i); }
}

fn shrinky_pointer(i: @@mut int) -> shrinky_pointer {
    shrinky_pointer {
        i: i
    }
}

fn main() {
    let my_total = @@mut 10;
    { let pt <- shrinky_pointer(my_total); assert (pt.look_at() == 10); }
    log(error, fmt!("my_total = %d", **my_total));
    assert (**my_total == 9);
}
