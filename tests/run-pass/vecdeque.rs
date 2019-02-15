use std::collections::VecDeque;

fn main() {
    let mut dst = VecDeque::new();
    dst.push_front(Box::new(1));
    dst.push_front(Box::new(2));
    dst.pop_back();

    let mut src = VecDeque::new();
    src.push_front(Box::new(2));
    dst.append(&mut src);
    for a in dst.iter() {
      assert_eq!(**a, 2);
    }

    // Regression test for Debug and Diaplay impl's
    println!("{:?} {:?}", dst, dst.iter());
    println!("{:?}", VecDeque::<u32>::new().iter());

    for a in dst {
      assert_eq!(*a, 2);
    }
}
