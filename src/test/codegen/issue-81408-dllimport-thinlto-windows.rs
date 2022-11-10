// compile-flags: -O -C lto=thin -C prefer-dynamic=no
// only-windows
// aux-build:static_dllimport_aux.rs

// Test that on Windows, when performing ThinLTO, we do not mark cross-crate static items with
// dllimport because lld does not fix the symbol names for us.

extern crate static_dllimport_aux;

// CHECK-LABEL: @{{.+}}CROSS_CRATE_STATIC_ITEM{{.+}} =
// CHECK-SAME: external local_unnamed_addr global %"{{.+}}AtomicPtr

pub fn main() {
    static_dllimport_aux::memrchr();
}
