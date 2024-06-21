struct Cat {
  meows : usize,
}

impl Cat {
    fn sleep(&self) { loop{} }
    fn meow(&self) {
      println!("Meow");
      meows += 1; //~ ERROR cannot find value `meows`
      sleep();     //~ ERROR cannot find function `sleep`
    }

}


 fn main() { }
