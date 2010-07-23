// error-pattern: Unsatisfied precondition constraint

fn test() {
    let vec[int] w;
    w.(5) = 0;
}

fn main() {
  test();
}