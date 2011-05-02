// -*- rust -*-

fn main() {
  if (!false) {
    check (true);
  } else {
    check (false);
  }

  if (!true) {
    check (false);
  } else {
    check (true);
  }
}
