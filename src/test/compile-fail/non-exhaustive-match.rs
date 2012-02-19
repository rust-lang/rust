enum t { a, b, }

fn main() {
    let x = a;
    alt x { b { } } //! ERROR non-exhaustive patterns
    alt true { //! ERROR non-exhaustive bool patterns
      true {}
    }
    alt @some(10) { //! ERROR non-exhaustive patterns
      @none {}
    }
    alt (2, 3, 4) { //! ERROR non-exhaustive literal patterns
      (_, _, 4) {}
    }
}
