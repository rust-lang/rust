class cat {
  priv {
    let mut meows : uint;
    fn sleep() { loop{} }
    fn meow() {
      error!{"Meow"};
      meows += 1u; //~ ERROR unresolved name
      sleep();     //~ ERROR unresolved name
    }
  }

  new(in_x : uint) { self.meows = in_x; }
}

 fn main() { }