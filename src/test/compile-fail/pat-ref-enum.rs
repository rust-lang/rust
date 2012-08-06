fn matcher(x: option<int>) {
    alt x {
      ref some(i) => {} //~ ERROR expected identifier, found enum pattern
      none => {}
    }
}

fn main() {}
