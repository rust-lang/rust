// repr currently doesn't support literals
#[repr("C")] //~ ERROR E0565
struct A {  }

fn main() {  }
