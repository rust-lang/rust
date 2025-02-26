//@ compile-flags: --diagnostic-width=60 -Zwrite-long-types-to-disk=yes
// The regex below normalizes the long type file name to make it suitable for compare-modes.
//@ normalize-stderr: "'\$TEST_BUILD_DIR/.*\.long-type-\d+.txt'" -> "'$$TEST_BUILD_DIR/$$FILE.long-type-hash.txt'"

type A = (String, String, String, String);
type B = (A, A, A, A);
type C = (B, B, B, B);
type D = (C, C, C, C);

trait Trait {}

fn require_trait<T: Trait>() {}

fn foo(x: D) {
    let _a = x;
    let _b = x; //~ ERROR use of moved value
}

fn main() {}
