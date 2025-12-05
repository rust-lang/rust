fn matcher(x: Option<isize>) {
    match x {
      ref Some(i) => {} //~ ERROR expected identifier, found enum pattern
      None => {}
    }
}

fn main() {}
