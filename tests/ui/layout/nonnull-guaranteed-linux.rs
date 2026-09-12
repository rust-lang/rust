//@ only-linux
#![feature(rustc_attrs)]
#![crate_type = "lib"]

// Check that various `#[rustc_nonnull_optimization_guaranteed]` types
// get their expected layouts inside `Option`s.

#[rustc_dump_layout(backend_repr)]
type OptFd = Option<std::os::unix::io::OwnedFd>;
//~^ ERROR: Scalar(i32 is ..)
