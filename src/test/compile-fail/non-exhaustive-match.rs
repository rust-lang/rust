enum t { a, b, }

fn main() {
    let x = a;
    alt x { b { } } //! ERROR non-exhaustive patterns
    alt true { //! ERROR non-exhaustive patterns
      true {}
    }
    alt @some(10) { //! ERROR non-exhaustive patterns
      @none {}
    }
    alt (2, 3, 4) { //! ERROR non-exhaustive patterns
      (_, _, 4) {}
    }
    alt (a, a) { //! ERROR non-exhaustive patterns
      (a, b) {}
      (b, a) {}
    }
    // This is exhaustive, though the algorithm got it wrong at one point
    alt (a, b) {
      (a, _) {}
      (_, a) {}
      (b, b) {}
    }
}
