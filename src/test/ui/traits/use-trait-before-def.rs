// run-pass
#![allow(non_camel_case_types)]

// Issue #1761

// pretty-expanded FIXME #23616

impl foo for isize { fn foo(&self) -> isize { 10 } }
trait foo { fn foo(&self) -> isize; }
pub fn main() {}
