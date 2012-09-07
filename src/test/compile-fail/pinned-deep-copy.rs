// error-pattern: copying a noncopyable value

struct r {
  i: @mut int,
  drop { *(self.i) = *(self.i) + 1; }
}

fn r(i: @mut int) -> r {
    r {
        i: i
    }
}

fn main() {
    let i = @mut 0;
    {
        // Can't do this copy
        let x = ~~~{y: r(i)};
        let z = x;
        log(debug, x);
    }
    log(error, *i);
}