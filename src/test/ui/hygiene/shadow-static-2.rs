// Check that a macro generated static does cause a shadowing error

macro_rules! h {
  () => {
    #[allow(non_upper_case_globals)]
    static x: usize = 3;
  }
}

h!();

fn main() {
  let x = 2;
//~^ ERROR let bindings cannot shadow statics
}
