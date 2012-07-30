fn main() {
    alt ~100 {
      ~x {
        debug!{"%?", x};
        assert x == 100;
      }
    }
}
