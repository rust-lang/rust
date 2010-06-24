type foo[T] = tag(arm(T));

fn altfoo[T](foo[T] f) {
  alt (f) {
    case (arm(x)) {}
  }
}

fn main() {}
