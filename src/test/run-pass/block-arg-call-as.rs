fn asBlock<F>(f: F) -> usize where F: FnOnce() -> usize {
   return f();
}

pub fn main() {
   let x = asBlock(|| 22);
   assert_eq!(x, 22);
}
