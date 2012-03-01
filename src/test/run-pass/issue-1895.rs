fn main() {
  let x = 1;
  let y = fn@[move x]() -> int {
             x
          }();
}

