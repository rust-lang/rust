// error-pattern:found `let` in restricted position

fn main() {
    alt true {
      {let} { }
    }
}
