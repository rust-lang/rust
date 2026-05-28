// this code used to cause an ICE

fn main() {
  let t = Err(0);
  match t {
    Some(k) => match k { //~ ERROR mismatched types
      a => println!("{}", a)
    },
    None => () //~ ERROR mismatched types
  }
}
