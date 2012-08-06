fn simple() {
    match ~true {
      ~true => { }
      _ => { fail; }
    }
}

fn main() {
    simple();
}