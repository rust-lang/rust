struct DropNoMethod;
impl Drop for DropNoMethod {} //~ ERROR not all trait items implemented, missing: `drop`

fn main() {}
