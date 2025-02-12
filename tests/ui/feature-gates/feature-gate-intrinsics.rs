extern "rust-intrinsic" {   //~ ERROR "rust-intrinsic" ABI is an implementation detail
    fn bar(); //~ ERROR unrecognized intrinsic function: `bar`
}

extern "rust-intrinsic" fn baz() {} //~ ERROR "rust-intrinsic" ABI is an implementation detail
//~^ ERROR intrinsic must be in

fn main() {}
