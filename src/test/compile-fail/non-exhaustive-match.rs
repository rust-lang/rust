enum t { a, b, }

fn main() {
    let x = a;
    match x { b => { } } //~ ERROR non-exhaustive patterns
    match true { //~ ERROR non-exhaustive patterns
      true => {}
    }
    match @some(10) { //~ ERROR non-exhaustive patterns
      @none => {}
    }
    match (2, 3, 4) { //~ ERROR non-exhaustive patterns
      (_, _, 4) => {}
    }
    match (a, a) { //~ ERROR non-exhaustive patterns
      (a, b) => {}
      (b, a) => {}
    }
    match a { //~ ERROR b not covered
      a => {}
    }
    // This is exhaustive, though the algorithm got it wrong at one point
    match (a, b) {
      (a, _) => {}
      (_, a) => {}
      (b, b) => {}
    }
}
