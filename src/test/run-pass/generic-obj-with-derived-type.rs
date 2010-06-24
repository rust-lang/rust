obj handle[T](T data) {
  fn get() -> T {
    ret data;
  }
}

fn main() {
  type rgb = tup(u8,u8,u8);
  let handle[rgb] h = handle[rgb](tup(u8(1), u8(2), u8(3)));
  log "constructed object";
  log h.get()._0;
  log h.get()._1;
  log h.get()._2;
  check (h.get()._0 == u8(1));
  check (h.get()._1 == u8(2));
  check (h.get()._2 == u8(3));
}
