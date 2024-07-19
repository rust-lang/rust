type A = S<Fn(i32)>;
type A = S<Fn(i32) + Send>;
type B = S<Fn(i32) -> i32>;
type C = S<Fn(i32) -> i32 + Send>;
