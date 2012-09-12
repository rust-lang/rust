// xfail-fast - check-fail fast doesn't under aux-build
// aux-build:issue2170lib.rs
extern mod issue2170lib;

fn main() {
   // let _ = issue2170lib::rsrc(2i32);
}
