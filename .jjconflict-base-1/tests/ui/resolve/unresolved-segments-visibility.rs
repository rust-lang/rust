// Check that we do not ICE due to unresolved segments in visibility path.
#![crate_type = "lib"]

extern crate alloc as b;

mod foo {
    mod bar {
        pub(in crate::b::string::String::newy) extern crate alloc as e;
        //~^ ERROR: cannot find module `String` in `string` [E0433]
        //~| NOTE: `String` is a struct, not a module
    }
}
