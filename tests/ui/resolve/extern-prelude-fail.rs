// compile-flags:--extern extern_prelude
// aux-build:extern-prelude.rs

// Extern prelude names are not available by absolute paths

fn main() {
    use extern_prelude::S; //~ ERROR unresolved import `extern_prelude`
    let s = ::extern_prelude::S; //~ ERROR failed to resolve
}
