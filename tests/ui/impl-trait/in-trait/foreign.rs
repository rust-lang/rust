// check-pass
// aux-build: rpitit.rs
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

extern crate rpitit;

fn main() {
    // Witness an RPITIT from another crate
    let () = <rpitit::Foreign as rpitit::Foo>::bar();
}
