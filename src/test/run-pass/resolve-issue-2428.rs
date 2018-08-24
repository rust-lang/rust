const foo: isize = 4 >> 1;
enum bs { thing = foo }
pub fn main() { assert_eq!(bs::thing as isize, foo); }
