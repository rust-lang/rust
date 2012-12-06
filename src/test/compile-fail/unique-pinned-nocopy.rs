// error-pattern: copying a noncopyable value

struct r {
  b:bool,
}

impl r : Drop {
    fn finalize(&self) {}
}

fn main() {
    let i = move ~r { b: true };
    let j = copy i;
    log(debug, i);
}
