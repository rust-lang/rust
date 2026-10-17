//@ run-rustfix
extern "systen" fn systen() {} //~ ERROR invalid ABI

fn main() {
    systen();
}
