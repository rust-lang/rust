// https://github.com/rust-lang/rust/issues/4736
// Ensure tuple structs cannot be constructed using struct field syntax.
struct NonCopyable(());

fn main() {
    let z = NonCopyable{ p: () }; //~ ERROR struct `NonCopyable` has no field named `p`
}
