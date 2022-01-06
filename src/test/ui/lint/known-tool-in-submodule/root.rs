// check-pass

#![cfg_attr(bootstrap, feature(register_tool))]
#![register_tool(tool)]

mod submodule;

fn main() {
    submodule::foo();
}
