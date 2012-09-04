fn main() {
  let mut i = 0u;
  loop {
    log(error, ~"a");
    i += 1u;
    if i == 10u {
      break;
    }
  }
  assert (i == 10u);
  let mut is_even = false;
  loop {
    if i == 21u {
        break;
    }
    log(error, ~"b");
    is_even = false;
    i += 1u;
    if i % 2u != 0u {
        loop;
    }
    is_even = true;
  }
  assert !is_even;
  loop {
    log(error, ~"c");
    if i == 22u {
        break;
    }
    is_even = false;
    i += 1u;
    if i % 2u != 0u {
        loop;
    }
    is_even = true;
  }
  assert is_even;
}