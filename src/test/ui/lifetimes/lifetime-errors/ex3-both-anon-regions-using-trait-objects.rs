fn foo(x:Box<Fn(&u8, &u8)> , y: Vec<&u8>, z: &u8) {
  y.push(z); //~ ERROR lifetime mismatch
}

fn main() { }
