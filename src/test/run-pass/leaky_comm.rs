// xfail-fast
// aux-build:test_comm.rs

use test_comm;

fn main() {
  let p = test_comm::port();
  
  match none::<int> {
      none => {}
      some(_)  =>{
  if test_comm::recv(p) == 0 {
      error!{"floop"};
  }
  else {
      error!{"bloop"};
  }
      }}
}