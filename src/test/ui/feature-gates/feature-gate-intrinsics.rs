extern "rust-intrinsic" {   //~ ERROR intrinsics are subject to change
    fn bar(); //~ ERROR unrecognized intrinsic function: `bar`
}

extern "rust-intrinsic" fn baz() {} //~ ERROR intrinsics are subject to change

fn main() {}
