fn matcher(x: option<int>) {
    match x {
      ref some(i) => {} //~ ERROR expected identifier, found enum pattern
      none => {}
    }
}

fn main() {}
