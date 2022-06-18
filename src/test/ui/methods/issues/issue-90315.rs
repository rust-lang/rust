fn main() {
  let arr = &[0,1,2,3];
  for _i in 0..arr.len().rev() { //~ERROR not an iterator
     // The above error used to say “the method `rev` exists for type `usize`”.
     // This regression test ensures it doesn't say that any more.
  }
}
