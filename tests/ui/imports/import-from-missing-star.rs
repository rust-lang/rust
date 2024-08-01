use spam::*; //~ ERROR unresolved import `spam` [E0432]

fn main() {
    // Expect these to pass because the compiler knows there's a failed `*` import that might have
    // caused it.
    ham();
    eggs();
    // Even this case, as we might have expected `spam::foo` to exist.
    foo::bar();
}
