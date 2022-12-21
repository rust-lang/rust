fn main() {
    (const || {})();
    //~^ ERROR: const closures are experimental
}
