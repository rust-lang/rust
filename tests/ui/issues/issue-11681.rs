// This tests verifies that unary structs and enum variants
// are treated as rvalues and their lifetime is not bounded to
// the static scope.

struct Test;

impl Drop for Test {
    fn drop (&mut self) {}
}

fn createTest<'a>() -> &'a Test {
  let testValue = &Test;
  return testValue; //~ ERROR cannot return value referencing temporary value
}


pub fn main() {
    createTest();
}
