struct cat {
  priv {
    mut meows : uint,
  }
}

priv impl cat {
    fn sleep() { loop{} }
    fn meow() {
      error!("Meow");
      meows += 1u; //~ ERROR unresolved name
      sleep();     //~ ERROR unresolved name
    }

}


 fn main() { }