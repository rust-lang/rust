// error-pattern: copying a noncopyable value

class r {
  let b:bool;
  new(b: bool) { self.b = b; }
  drop {}
}

fn main() {
    let i <- ~r(true);
    let j = i;
    log(debug, i);
}