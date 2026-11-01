trait Hello {
  fn example(&self, input: &i32);
}

struct Test1(i32);

impl Hello for Test1 {
  fn example(&self, input: &i32) {
      *input = self.0; //~ ERROR cannot assign
  }
}

struct Test2(i32);

impl Hello for Test2 {
  fn example(&self, input: &i32) {
    self.0 += *input; //~ ERROR cannot assign
  }
}

fn main() { }
