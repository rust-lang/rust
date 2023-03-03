// check-pass
// aux-build: rpitit.rs
// ignore-compare-mode-lower-impl-trait-in-trait-to-assoc-ty

extern crate rpitit;

fn main() {
    // Witness an RPITIT from another crate
    let () = <rpitit::Foreign as rpitit::Foo>::bar();
}
