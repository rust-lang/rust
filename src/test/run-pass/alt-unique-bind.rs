fn main() {
    match ~100 {
      ~x => {
        debug!("%?", x);
        assert x == 100;
      }
    }
}
