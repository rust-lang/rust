// xfail-test
// runs forever for some reason -- investigating
fn main() {
  let i = 0u;
  loop {
    log(error, "a");
    i += 1u;
    if i == 10u {
      break;
    }
  }
  assert (i == 10u);
  let is_even = false;
  loop {
    log(error, "b");
    is_even = false;
    i += 1u;
    if i % 2u != 0u {
        cont;
    }
    is_even = true;
    if i == 21u {
        break;
    }
  }
  assert !is_even;
  loop {
    log(error, "c");
    is_even = false;
    if i == 22u {
        break;
    }
    i += 1u;
    if i % 2u != 0u {
        cont;
    }
    is_even = true;
  }
  assert is_even;
}