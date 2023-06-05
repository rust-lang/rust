fn main() {}

// unconst and bad, will thus error in miri
const X: bool = unsafe { &1 as *const i32 == &2 as *const i32 }; //~ ERROR pointers cannot
// unconst and bad, will thus error in miri
const X2: bool = unsafe { 42 as *const i32 == 43 as *const i32 }; //~ ERROR pointers cannot
