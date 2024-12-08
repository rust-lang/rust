fn a<F>(_f: F) -> ()
where
  F: FnOnce() -> (),
{
}
fn main() {
  a(|| {
    #[allow(irrefutable_let_patterns)]
    while let _ = 0 {
      break;
    }
  });
}