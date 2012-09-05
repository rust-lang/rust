// xfail-fast
// aux-build:anon-extern-mod-cross-crate-1.rs
use anonexternmod;

use anonexternmod::*;

fn main() {
  last_os_error();
}
