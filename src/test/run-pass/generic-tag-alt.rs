type foo[T] = tag(arm(T));

fn altfoo[T](foo[T] f) {
  auto hit = false;
  alt (f) {
    case (arm[T](x)) {
      log "in arm";
      hit = true;
    }
  }
  check (hit);
}

fn main() {
  altfoo[int](arm[int](10));
}
