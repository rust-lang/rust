// error-pattern:found `let` in restricted position

fn main() {
    match true {
      {let} { }
    }
}
