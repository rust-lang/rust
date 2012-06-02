// error-pattern: copying a noncopyable value

class r {
  new(_i:int) {}
  drop {}
}

fn main() {
    // This can't make sense as it would copy the classes
    let i <- [r(0)];
    let j <- [r(1)];
    let k = i + j;
    log(debug, j);
}
