// error-pattern: copying a noncopyable value

struct r {
  b:bool,
}

impl r : Drop {
    fn finalize() {}
}

fn main() {
    let i = move ~r { b: true };
    let j = i;
    log(debug, i);
}
