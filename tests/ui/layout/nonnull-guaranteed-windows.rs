//@ only-windows
//@ only-64bit
#![feature(rustc_attrs)]
#![crate_type = "lib"]

// Check that various `#[rustc_nonnull_optimization_guaranteed]` types
// get their expected layouts inside `Option`s.

#[rustc_dump_layout(backend_repr)]
type OptSocket = Option<std::os::windows::io::OwnedSocket>;
//~^ ERROR: Scalar(u64 is ..)
