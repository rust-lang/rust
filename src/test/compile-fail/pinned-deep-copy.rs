// error-pattern: copying a noncopyable value

struct r {
  let i: @mut int;
  new(i: @mut int) { self.i = i; }
  drop { *(self.i) = *(self.i) + 1; }
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