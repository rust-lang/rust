enum t { a, b, }

fn main() {
    let x = a;
    match x { b => { } } //~ ERROR non-exhaustive patterns
    match true { //~ ERROR non-exhaustive patterns
      true => {}
    }
    match @Some(10) { //~ ERROR non-exhaustive patterns
      @None => {}
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
    match ~[Some(42), None, Some(21)] { //~ ERROR non-exhaustive patterns: vectors of length 0 not covered
        [Some(*), None, ..tail] => {}
        [Some(*), Some(*), ..tail] => {}
        [None] => {}
    }
    match ~[1] {
        [_, ..tail] => (),
        [] => ()
    }
    match ~[0.5] { //~ ERROR non-exhaustive patterns: vectors of length 4 not covered
        [0.1, 0.2, 0.3] => (),
        [0.1, 0.2] => (),
        [0.1] => (),
        [] => ()
    }
    match ~[Some(42), None, Some(21)] {
        [Some(*), None, ..tail] => {}
        [Some(*), Some(*), ..tail] => {}
        [None, None, ..tail] => {}
        [None, Some(*), ..tail] => {}
        [Some(_)] => {}
        [None] => {}
        [] => {}
    }
}
