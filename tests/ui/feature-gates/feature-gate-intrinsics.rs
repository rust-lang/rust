extern "rust-intrinsic" {   //~ ERROR intrinsics are subject to change
    fn bar(); //~ ERROR unrecognized intrinsic function: `bar`
}

extern "rust-intrinsic" fn baz() {} //~ ERROR intrinsics are subject to change
//~^ ERROR intrinsic must be in
//~| ERROR unrecognized intrinsic function: `baz`

fn main() {}
