struct S<I: Iterator>(I);
struct T(S<u8>);
fn main() {}
