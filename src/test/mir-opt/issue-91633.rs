// compile-flags: -Z mir-opt-level=0
// EMIT_MIR issue_91633.hey.mir_map.0.mir
fn hey<T> (it: &[T])
 where
     [T] : std::ops::Index<usize>,
 {
     let _ = &it[0];
 }

// EMIT_MIR issue_91633.bar.mir_map.0.mir
fn bar<T> (it: Box<[T]>)
 where
     [T] : std::ops::Index<usize>,
 {
     let _ = it[0];
 }

// EMIT_MIR issue_91633.fun.mir_map.0.mir
fn fun<T> (it: &[T]) -> &T
 {
     let f = &it[0];
     f
 }

// EMIT_MIR issue_91633.foo.mir_map.0.mir
fn foo<T: Clone> (it: Box<[T]>) -> T
 {
     let f = it[0].clone();
     f
 }
 fn main(){}
