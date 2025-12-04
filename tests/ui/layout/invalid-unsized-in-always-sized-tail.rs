// issue: rust-lang/rust#126939

//! This used to ICE, because the layout algorithm did not check for unsized types
//! in the struct tail of always-sized types (i.e. those that cannot be unsized)
//! and incorrectly returned an unsized layout.

struct MySlice<T>(T);
type MySliceBool = MySlice<[bool]>;

struct P2 {
    b: MySliceBool,
    //~^ ERROR: the size for values of type `[bool]` cannot be known at compilation time
}

static CHECK: () = assert!(align_of::<P2>() == 1);
//~^ ERROR the type `MySlice<[bool]>` has an unknown layout

fn main() {}
