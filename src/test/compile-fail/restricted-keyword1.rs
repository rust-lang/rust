// error-pattern:found `fail` in restricted position

fn main() {
    match true {
      {fail} { }
    }
}
