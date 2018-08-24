extern "rust-intrinsic" {   //~ ERROR intrinsics are subject to change
    fn bar();
}

extern "rust-intrinsic" fn baz() {  //~ ERROR intrinsics are subject to change
}

fn main() {
}
