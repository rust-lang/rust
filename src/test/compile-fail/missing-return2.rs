// error-pattern: return

fn f() -> int {
  // Make sure typestate doesn't interpret this alt expression
  // as the function result
  alt (true) {
    case (true) {
    }
  }
}

fn main() {
}
