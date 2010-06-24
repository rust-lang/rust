type tree = tag(nil(), node(@tree, @tree, int));

fn item_check(&tree t) -> int {
  alt (t) {
    case (nil()) {
      ret 0;
    }
    case (node(@tree left, @tree right, int item)) {
      ret item + item_check(left) - item_check(right);
    }
  }
}

fn main() {
}