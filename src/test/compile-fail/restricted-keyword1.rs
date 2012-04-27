// error-pattern:found `let` in binding position

fn main() {
    alt true {
      {let} { }
    }
}
