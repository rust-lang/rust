// return -> return
// mod -> module
// match -> match

fn main() {
}

mod foo {
    #[legacy_exports];
}

fn bar() -> int {
    match 0 {
      _ => { 0 }
    }
}