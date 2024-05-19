trait Tr {}
type Huh<T> where T: Tr = isize; //~ ERROR type parameter `T` is never used
fn main() {}
