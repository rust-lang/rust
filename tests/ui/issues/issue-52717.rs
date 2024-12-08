enum A {
  A {
    foo: usize,
  }
}

fn main() {
  let x = A::A { foo: 3 };
  match x {
    A::A { fob } => { println!("{}", fob); }
//~^ ERROR does not have a field named `fob`
  }
}
