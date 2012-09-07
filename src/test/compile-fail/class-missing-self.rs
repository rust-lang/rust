struct cat {
  priv {
    mut meows : uint,
    fn sleep() { loop{} }
    fn meow() {
      error!("Meow");
      meows += 1u; //~ ERROR unresolved name
      sleep();     //~ ERROR unresolved name
    }
  }

}


 fn main() { }