struct DropNoMethod;
impl Drop for DropNoMethod {} //~ ERROR not all trait items implemented, missing one of: `drop`, `pin_drop`

fn main() {}
