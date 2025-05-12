//@ check-pass

#![deny(unused_attributes)] // c.f #35584

mod auxiliary {
    #[cfg_attr(any(), path = "nonexistent_file.rs")] pub mod namespaced_enums;
    #[cfg_attr(all(), path = "namespaced_enums.rs")] pub mod nonexistent_file;
}

fn main() {
    let _ = auxiliary::namespaced_enums::Foo::A;
    let _ = auxiliary::nonexistent_file::Foo::A;
}
