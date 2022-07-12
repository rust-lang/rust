// build-pass

// Check that a macro doesn't generate an error when a let binding shadows a
// static defined outside the macro.

macro_rules! h {
  () => {
    let x = 2;
  }
}

#[allow(non_upper_case_globals)]
static x: usize = 3;

fn main() {
  h!();
}
