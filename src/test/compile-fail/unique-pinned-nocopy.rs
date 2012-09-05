// error-pattern: copying a noncopyable value

struct r {
  let b:bool;
  drop {}
}

fn main() {
    let i <- ~r { b: true };
    let j = i;
    log(debug, i);
}