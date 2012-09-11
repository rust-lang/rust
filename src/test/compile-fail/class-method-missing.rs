// error-pattern:missing method `eat`
trait animal {
  fn eat();
}

struct cat {
  meows: uint,
}

impl cat : animal {
}

fn cat(in_x : uint) -> cat {
    cat {
        meows: in_x
    }
}

fn main() {
  let nyan = cat(0u);
}