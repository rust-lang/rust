// Check that we do not ICE due to unresolved segments in visibility path.
#![crate_type = "lib"]

extern crate alloc as b;

mod foo {
    mod bar {
        pub(in crate::b::string::String::newy) extern crate alloc as e;
        //~^ ERROR failed to resolve: `String` is a struct, not a module [E0433]
    }
}
