//@ known-bug: #140683
impl T {
#[core::contracts::ensures]
  fn b() { (loop) }
}
