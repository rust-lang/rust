//@ revisions: all strong basic none missing
//@ assembly-output: emit-asm
//@ only-windows
//@ only-msvc
//@ ignore-64bit 64-bit table based SEH has slightly different behaviors than classic SEH
//@ [all] compile-flags: -Z stack-protector=all
//@ [strong] compile-flags: -Z stack-protector=strong
//@ [basic] compile-flags: -Z stack-protector=basic
//@ [none] compile-flags: -Z stack-protector=none
//@ compile-flags: -C opt-level=2 -Z merge-functions=disabled

#![crate_type = "lib"]
#![allow(internal_features)]
#![feature(unsized_fn_params)]

// CHECK-LABEL: local_string_addr_taken
#[no_mangle]
pub fn local_string_addr_taken(f: fn(&String)) {
    let x = String::new();
    f(&x);

    // Taking the address of the local variable `x` leads to stack smash
    // protection with the `strong` heuristic, but not with the `basic`
    // heuristic. It does not matter that the reference is not mut.
    //
    // An interesting note is that a similar function in C++ *would* be
    // protected by the `basic` heuristic, because `std::string` has a char
    // array internally as a small object optimization:
    // ```
    // cat <<EOF | clang++ -O2 -fstack-protector -S -x c++ - -o - | grep stack_chk
    // #include <string>
    // void f(void (*g)(const std::string&)) {
    //     std::string x;
    //     g(x);
    // }
    // EOF
    // ```
    //

    // all: __security_check_cookie
    // strong-NOT: __security_check_cookie
    // basic-NOT: __security_check_cookie
    // none-NOT: __security_check_cookie
    // missing-NOT: __security_check_cookie
}
