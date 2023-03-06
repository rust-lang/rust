// This tests verifies that unary structs and enum variants
// are treated as rvalues and their lifetime is not bounded to
// the static scope.

struct Test;

impl Drop for Test {
    fn drop(&mut self) {}
}

fn create_test<'a>() -> &'a Test {
    let test_value = &Test;
    return test_value; //~ ERROR cannot return value referencing temporary value
}

pub fn main() {
    create_test();
}
