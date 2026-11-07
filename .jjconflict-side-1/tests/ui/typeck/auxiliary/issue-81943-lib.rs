pub fn g(t: i32) -> i32 { t }
// This function imitates `dbg!` so that future changes
// to its macro definition won't make this test a dud.
#[macro_export]
macro_rules! d {
  ($e:expr) => { match $e { x => { $crate::g(x) } } }
}
