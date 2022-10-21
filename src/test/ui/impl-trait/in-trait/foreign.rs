// check-pass
// aux-build: rpitit.rs

extern crate rpitit;

fn main() {
    // Witness an RPITIT from another crate
    let () = <rpitit::Foreign as rpitit::Foo>::bar();
}
