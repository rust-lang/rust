tag tree {
  nil;
  node(@tree, @tree, int);
}

fn item_check(@tree t) -> int {
  alt (*t) {
    case (nil) {
      ret 0;
    }
    case (node(?left, ?right, ?item)) {
      ret item + item_check(left) - item_check(right);
    }
  }
}

fn main() {
}