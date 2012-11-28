// error-pattern: copying a noncopyable value

struct my_resource {
  x: int,
}

impl my_resource : Drop {
    fn finalize(&self) {
        log(error, self.x);
    }
}

fn my_resource(x: int) -> my_resource {
    my_resource {
        x: x
    }
}

fn main() {
    {
        let a = {x: 0, y: my_resource(20)};
        let b = {x: 2,.. a};
        log(error, (a, b));
    }
}
