//@ run-rustfix
mod a_module {
  pub struct ActiveType;
}

#[deprecated]
//~^ ERROR this `#[deprecated]` annotation has no effect
pub use a_module::ActiveType;

fn main() {
}
