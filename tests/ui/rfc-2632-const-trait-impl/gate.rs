// gate-test-const_closures
fn main() {
    (const || {})();
    //~^ ERROR: const closures are experimental
}
