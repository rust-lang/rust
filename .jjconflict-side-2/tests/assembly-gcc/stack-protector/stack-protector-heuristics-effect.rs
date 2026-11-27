//@ revisions: all strong basic none missing
//@ assembly-output: emit-asm
//@ ignore-apple slightly different policy on stack protection of arrays
//@ ignore-msvc stack check code uses different function names
//@ ignore-nvptx64 stack protector is not supported
//@ ignore-wasm32-unknown-unknown
//@ [all] compile-flags: -Z stack-protector=all
//@ [strong] compile-flags: -Z stack-protector=strong
//@ [basic] compile-flags: -Z stack-protector=basic
//@ [none] compile-flags: -Z stack-protector=none
//@ compile-flags: -C opt-level=2 -Z merge-functions=disabled

#![crate_type = "lib"]
#![allow(internal_features)]
#![feature(unsized_fn_params)]

// CHECK-LABEL: "emptyfn"
#[no_mangle]
pub fn emptyfn() {
    // all: __stack_chk_fail
    // strong-NOT: __stack_chk_fail
    // basic-NOT: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

// CHECK-LABEL: "array_char"
#[no_mangle]
pub fn array_char(f: fn(*const char)) {
    let a = ['c'; 1];
    let b = ['d'; 3];
    let c = ['e'; 15];

    f(&a as *const _);
    f(&b as *const _);
    f(&c as *const _);

    // all: __stack_chk_fail
    // strong: __stack_chk_fail
    // basic: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

// CHECK-LABEL: "array_u8_1"
#[no_mangle]
pub fn array_u8_1(f: fn(*const u8)) {
    let a = [0u8; 1];
    f(&a as *const _);

    // The 'strong' heuristic adds stack protection to functions with local
    // array variables regardless of their size.

    // all: __stack_chk_fail
    // strong: __stack_chk_fail
    // basic-NOT: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

// CHECK-LABEL: "array_u8_small"
#[no_mangle]
pub fn array_u8_small(f: fn(*const u8)) {
    let a = [0u8; 2];
    let b = [0u8; 7];
    f(&a as *const _);
    f(&b as *const _);

    // Small arrays do not lead to stack protection by the 'basic' heuristic.

    // all: __stack_chk_fail
    // strong: __stack_chk_fail
    // basic-NOT: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

// CHECK-LABEL: "array_u8_large"
#[no_mangle]
pub fn array_u8_large(f: fn(*const u8)) {
    let a = [0u8; 9];
    f(&a as *const _);

    // Since `a` is a byte array with size greater than 8, the basic heuristic
    // will also protect this function.

    // all: __stack_chk_fail
    // strong: __stack_chk_fail
    // basic: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

#[derive(Copy, Clone)]
pub struct ByteSizedNewtype(u8);

// CHECK-LABEL: "array_bytesizednewtype_9"
#[no_mangle]
pub fn array_bytesizednewtype_9(f: fn(*const ByteSizedNewtype)) {
    let a = [ByteSizedNewtype(0); 9];
    f(&a as *const _);

    // Since `a` is a byte array in the GCC output, the basic heuristic will
    // also protect this function.

    // all: __stack_chk_fail
    // strong: __stack_chk_fail
    // basic: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

// CHECK-LABEL: "local_var_addr_used_indirectly"
#[no_mangle]
pub fn local_var_addr_used_indirectly(f: fn(bool)) {
    let a = 5;
    let a_addr = &a as *const _ as usize;
    f(a_addr & 0x10 == 0);

    // This function takes the address of a local variable taken. Although this
    // address is never used as a way to refer to stack memory, the `strong`
    // heuristic adds stack smash protection. This is also the case in C++:
    // ```
    // cat << EOF | g++ -O2 -fstack-protector-strong -S -x c++ - -o - | grep stack_chk
    // #include <cstdint>
    // void f(void (*g)(bool)) {
    //     int32_t x;
    //     g((reinterpret_cast<uintptr_t>(&x) & 0x10U) == 0);
    // }
    // EOF
    // ```

    // all: __stack_chk_fail
    // strong: __stack_chk_fail
    // basic-NOT: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

// CHECK-LABEL: "local_string_addr_taken"
#[no_mangle]
pub fn local_string_addr_taken(f: fn(&String)) {
    let x = String::new();
    f(&x);

    // Taking the address of the local variable `x` leads to stack smash
    // protection. It does not matter that the reference is not mut.

    // all: __stack_chk_fail
    // strong: __stack_chk_fail
    // basic: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

pub trait SelfByRef {
    fn f(&self) -> i32;
}

impl SelfByRef for i32 {
    fn f(&self) -> i32 {
        return self + 1;
    }
}

// CHECK-LABEL: "local_var_addr_taken_used_locally_only"
#[no_mangle]
pub fn local_var_addr_taken_used_locally_only(factory: fn() -> i32, sink: fn(i32)) {
    let x = factory();
    let g = x.f();
    sink(g);

    // Even though the local variable conceptually has its address taken, as
    // it's passed by reference to the trait function, the use of the reference
    // is easily inlined. There is therefore no stack smash protection even with
    // the `strong` heuristic.

    // all: __stack_chk_fail
    // strong-NOT: __stack_chk_fail
    // basic-NOT: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

pub struct Gigastruct {
    does: u64,
    not: u64,
    have: u64,
    array: u64,
    members: u64,
}

// CHECK-LABEL: "local_large_var_moved"
#[no_mangle]
pub fn local_large_var_moved(f: fn(Gigastruct)) {
    let x = Gigastruct { does: 0, not: 1, have: 2, array: 3, members: 4 };
    f(x);

    // Even though the local variable conceptually doesn't have its address
    // taken, it's so large that the "move" is implemented with a reference to a
    // stack-local variable in the ABI. Consequently, this function *is*
    // protected. This is also the case for rvalue-references in C++,
    // regardless of struct size:
    // ```
    // cat <<EOF | g++ -O2 -fstack-protector-strong -S -x c++ - -o - | grep stack_chk
    // #include <cstdint>
    // #include <utility>
    // void f(void (*g)(uint64_t&&)) {
    //     uint64_t x;
    //     g(std::move(x));
    // }
    // EOF
    // ```

    // all: __stack_chk_fail
    // strong: __stack_chk_fail
    // basic: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

// CHECK-LABEL: "local_large_var_cloned"
#[no_mangle]
pub fn local_large_var_cloned(f: fn(Gigastruct)) {
    f(Gigastruct { does: 0, not: 1, have: 2, array: 3, members: 4 });

    // A new instance of `Gigastruct` is passed to `f()`, without any apparent
    // connection to this stack frame. Still, since instances of `Gigastruct`
    // are sufficiently large, it is allocated in the caller stack frame and
    // passed as a pointer. As such, this function is *also* protected, just
    // like `local_large_var_moved`.
    //
    // This matches clang++ behavior, but not g++ behavior.
    //
    // In any case, both options are fine from a specification point of view, there
    // is no "user-accessible pointer", and there is no strong reason to avoid generating
    // a canary in this case, since it doesn't seem to be one of the performance-critical
    // cases in which avoiding generating a canary is important, so it seems that
    // rustc should keep the clang-like behavior of generating a canary here.
    //
    // ```
    // cat <<EOF | g++ -O2 -fstack-protector-strong -S -x c++ - -o - | grep stack_chk
    // #include <cstdint>
    // #include <utility>
    // struct Gigastruct { uint64_t a, b, c, d, e; };
    // void f(void (*g)(Gigastruct)) {
    //     g(Gigastruct{});
    // }
    // EOF
    // ```

    // all: __stack_chk_fail
    // strong: __stack_chk_fail
    // basic: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

extern "C" {
    // Difference between LLVM and GCC: LLVM will not generate stack protection
    // for "external" calls to alloca, but gcc will. See the matching test for
    // stack-protector-heuristics-effect under assembly-llvm.
    //
    // This is a difference in heuristics and therefore fine.
    //
    // Check that rustc_codegen_gcc matches gcc behavior.

    fn alloca(size: usize) -> *mut ();
}

// CHECK-LABEL: "alloca_small_compile_time_constant_arg"
#[no_mangle]
pub fn alloca_small_compile_time_constant_arg(f: fn(*mut ())) {
    f(unsafe { alloca(8) });

    // all: __stack_chk_fail
    // strong: __stack_chk_fail
    // basic: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

// CHECK-LABEL: "alloca_large_compile_time_constant_arg"
#[no_mangle]
pub fn alloca_large_compile_time_constant_arg(f: fn(*mut ())) {
    f(unsafe { alloca(9) });

    // all: __stack_chk_fail
    // strong: __stack_chk_fail
    // basic: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

// CHECK-LABEL: "alloca_dynamic_arg"
#[no_mangle]
pub fn alloca_dynamic_arg(f: fn(*mut ()), n: usize) {
    f(unsafe { alloca(n) });

    // all: __stack_chk_fail
    // strong: __stack_chk_fail
    // basic: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

// rustc can currently (as of 1.98) not generate variable-sized allocas, except for
// variable-sized scalable vector types, so their interaction with stack-protector
// does not need to be tested.
