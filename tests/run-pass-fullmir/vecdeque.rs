use std::collections::VecDeque;

fn main() {
    let mut dst = VecDeque::new();
    dst.push_front(Box::new(1));
    dst.push_front(Box::new(2));
    dst.pop_back();

    let mut src = VecDeque::new();
    src.push_front(Box::new(2));
    dst.append(&mut src);
    for a in dst {
      assert_eq!(*a, 2);
    }
}
