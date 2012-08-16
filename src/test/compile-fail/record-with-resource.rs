// error-pattern: copying a noncopyable value

struct my_resource {
  let x: int;
  new(x: int) { self.x = x; }
  drop { log(error, self.x); }
}

fn main() {
    {
        let a = {x: 0, y: my_resource(20)};
        let b = {x: 2 with a};
        log(error, (a, b));
    }
}
