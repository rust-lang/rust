// error-pattern: copying a noncopyable value

resource r(b: bool) {
}

fn main() {
    let i <- ~r(true);
    let j = i;
    log(debug, i);
}