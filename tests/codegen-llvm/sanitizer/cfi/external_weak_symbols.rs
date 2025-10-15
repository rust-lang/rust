// Verifies that type metadata identifiers for for weakly-linked symbols are
// emitted correctly.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clinker-plugin-lto -Copt-level=0 -Zsanitizer=cfi -Ctarget-feature=-crt-static -C unsafe-allow-abi-mismatch=sanitizer
#![crate_type = "bin"]
#![feature(linkage)]

unsafe extern "C" {
    #[linkage = "extern_weak"]
    static FOO: Option<unsafe extern "C" fn(f64) -> ()>;
}
// CHECK: @_rust_extern_with_linkage_{{.*}}_FOO = internal global ptr @FOO

fn main() {
    unsafe {
        if let Some(method) = FOO {
            method(4.2);
            // CHECK: call i1 @llvm.type.test(ptr {{%method|%0}}, metadata !"_ZTSFvdE")
        }
    }
}

// CHECK: declare !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}} extern_weak void @FOO(double) unnamed_addr #{{[0-9]+}}
