// error-pattern:get called on error result: ~"kitty"
fn main() {
  log(error, result::get(&result::Err::<int,~str>(~"kitty")));
}
