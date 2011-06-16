
/* 
   This is a test case for Issue 507.

   https://github.com/graydon/rust/issues/507
*/

use std;

import std::task::join;

fn grandchild(chan[int] c) {
  c <| 42;
}

fn child(chan[int] c) {
  auto _grandchild = spawn grandchild(c);
  join(_grandchild);
}

fn main() {
  let port[int] p = port();

  auto _child = spawn child(chan(p));
  
  let int x;
  p |> x;

  log x;

  assert(x == 42);

  join(_child);
}
