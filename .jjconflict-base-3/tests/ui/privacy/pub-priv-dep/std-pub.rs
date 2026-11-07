// The 'std' crates should always be implicitly public,
// without having to pass any compiler arguments

//@ run-pass

#![deny(exported_private_dependencies)]

pub struct PublicType {
    pub field: Option<u8>
}

fn main() {}
