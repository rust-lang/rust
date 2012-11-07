// xfail-test
fn main() {
  let s = shell!( uname -a );
  log(debug, s);
}
