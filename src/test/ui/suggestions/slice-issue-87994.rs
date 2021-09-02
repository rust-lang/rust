fn main() {
  let v = vec![1i32, 2, 3];
  for _ in v[1..] {
    //~^ ERROR [i32]` is not an iterator [E0277]
    //~^^ ERROR known at compilation time
  }
}
