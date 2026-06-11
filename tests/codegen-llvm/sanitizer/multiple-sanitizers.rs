// Verifies that multiple compatible sanitizers (CFI + SafeStack) can be enabled together
// and their target modifiers accumulate instead of overwriting.
//
//@ only-x86_64
//@ only-linux
//@ compile-flags: -Zsanitizer=cfi -Zsanitizer=safestack -Clto -Ccodegen-units=1 -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]

// CHECK: ; Function Attrs:{{.*}}safestack
// CHECK: define{{.*}}foo{{.*}}!type
pub fn foo(f: fn(i32) -> i32, arg: i32) -> i32 {
    f(arg)
}

// CHECK: attributes #0 = {{.*}}safestack{{.*}}
