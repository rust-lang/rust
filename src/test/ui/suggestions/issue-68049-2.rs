trait Hello {
  fn example(&self, input: &i32); // should suggest here
}

struct Test1(i32);

impl Hello for Test1 {
  fn example(&self, input: &i32) { // should not suggest here
      *input = self.0;
  }
}

struct Test2(i32);

impl Hello for Test2 {
  fn example(&self, input: &i32) { // should not suggest here
    self.0 += *input;
  }
}

fn main() { }
