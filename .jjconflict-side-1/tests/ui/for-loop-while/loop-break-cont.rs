//@ run-pass
pub fn main() {
  let mut i = 0_usize;
  loop {
    println!("a");
    i += 1_usize;
    if i == 10_usize {
      break;
    }
  }
  assert_eq!(i, 10_usize);
  let mut is_even = false;
  loop {
    if i == 21_usize {
        break;
    }
    println!("b");
    is_even = false;
    i += 1_usize;
    if i % 2_usize != 0_usize {
        continue;
    }
    is_even = true;
  }
  assert!(!is_even);
  loop {
    println!("c");
    if i == 22_usize {
        break;
    }
    is_even = false;
    i += 1_usize;
    if i % 2_usize != 0_usize {
        continue;
    }
    is_even = true;
  }
  assert!(is_even);
}
