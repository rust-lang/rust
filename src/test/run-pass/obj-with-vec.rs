fn main() {

  obj buf(vec[u8] data) {
    fn get(int i) -> u8 {
      ret data.(i);
    }
  }
  auto b = buf(vec(u8(1), u8(2), u8(3)));
  log b.get(1);
  check (b.get(1) == u8(2));
}