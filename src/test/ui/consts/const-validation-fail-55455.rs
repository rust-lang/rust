// compile-pass

struct This<T>(T);

const C: This<Option<&i32>> = This(Some(&1));

fn main() {}
