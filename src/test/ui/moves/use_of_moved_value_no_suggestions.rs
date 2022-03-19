// No suggestions? :(

// In the future, we may want to suggest deriving `Clone, Copy` for `No` (and then adding `T: Copy`)
struct No;
fn move_non_clone_non_copy<T>(t: (T, No)) {
    [t, t]; //~ use of moved value: `t`
}

fn main() {}
