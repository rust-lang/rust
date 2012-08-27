// xfail-fast
// aux-build:test_comm.rs

use test_comm;

fn main() {
  let p = test_comm::port();
  
  match None::<int> {
      None => {}
      Some(_) => {
  if 0 == test_comm::recv(p) {
      error!("floop");
  }
  else {
      error!("bloop");
  }
      }}
}
