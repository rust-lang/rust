// revisions: all strong basic none missing
// assembly-output: emit-asm
// ignore-macos slightly different policy on stack protection of arrays
// ignore-windows stack check code uses different function names
// ignore-nvptx64 stack protector is not supported
// [all] compile-flags: -Z stack-protector=all
// [strong] compile-flags: -Z stack-protector=strong
// [basic] compile-flags: -Z stack-protector=basic
// [none] compile-flags: -Z stack-protector=none
// compile-flags: -C opt-level=2 -Z merge-functions=disabled

#![crate_type = "lib"]

#![allow(incomplete_features)]

#![feature(unsized_locals, unsized_fn_params)]


// CHECK-LABEL: emptyfn:
#[no_mangle]
pub fn emptyfn() {
    // all: __stack_chk_fail
    // strong-NOT: __stack_chk_fail
    // basic-NOT: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

// CHECK-LABEL: array_char
#[no_mangle]
pub fn array_char(f: fn(*const char)) {
    let a = ['c'; 1];
    let b = ['d'; 3];
    let c = ['e'; 15];

    f(&a as *const _);
    f(&b as *const _);
    f(&c as *const _);

    // Any type of local array variable leads to stack protection with the
    // "strong" heuristic. The 'basic' heuristic only adds stack protection to
    // functions with local array variables of a byte-sized type, however. Since
    // 'char' is 4 bytes in Rust, this function is not protected by the 'basic'
    // heuristic
    //
    // (This test *also* takes the address of the local stack variables. We
    // cannot know that this isn't what triggers the `strong` heuristic.
    // However, the test strategy of passing the address of a stack array to an
    // external function is sufficient to trigger the `basic` heuristic (see
    // test `array_u8_large()`). Since the `basic` heuristic only checks for the
    // presence of stack-local array variables, we can be confident that this
    // test also captures this part of the `strong` heuristic specification.)

    // all: __stack_chk_fail
    // strong: __stack_chk_fail
    // basic-NOT: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

// CHECK-LABEL: array_u8_1
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

// CHECK-LABEL: array_u8_small:
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

// CHECK-LABEL: array_u8_large:
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

// CHECK-LABEL: array_bytesizednewtype_9:
#[no_mangle]
pub fn array_bytesizednewtype_9(f: fn(*const ByteSizedNewtype)) {
    let a = [ByteSizedNewtype(0); 9];
    f(&a as *const _);

    // Since `a` is a byte array in the LLVM output, the basic heuristic will
    // also protect this function.

    // all: __stack_chk_fail
    // strong: __stack_chk_fail
    // basic: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

// CHECK-LABEL: local_var_addr_used_indirectly
#[no_mangle]
pub fn local_var_addr_used_indirectly(f: fn(bool)) {
    let a = 5;
    let a_addr = &a as *const _ as usize;
    f(a_addr & 0x10 == 0);

    // This function takes the address of a local variable taken. Although this
    // address is never used as a way to refer to stack memory, the `strong`
    // heuristic adds stack smash protection. This is also the case in C++:
    // ```
    // cat << EOF | clang++ -O2 -fstack-protector-strong -S -x c++ - -o - | grep stack_chk
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

    // all: __stack_chk_fail
    // strong: __stack_chk_fail
    // basic-NOT: __stack_chk_fail
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

// CHECK-LABEL: local_var_addr_taken_used_locally_only
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
    members: u64
}

// CHECK-LABEL: local_large_var_moved
#[no_mangle]
pub fn local_large_var_moved(f: fn(Gigastruct)) {
    let x = Gigastruct { does: 0, not: 1, have: 2, array: 3, members: 4 };
    f(x);

    // Even though the local variable conceptually doesn't have its address
    // taken, it's so large that the "move" is implemented with a reference to a
    // stack-local variable in the ABI. Consequently, this function *is*
    // protected by the `strong` heuristic. This is also the case for
    // rvalue-references in C++, regardless of struct size:
    // ```
    // cat <<EOF | clang++ -O2 -fstack-protector-strong -S -x c++ - -o - | grep stack_chk
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
    // basic-NOT: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

// CHECK-LABEL: local_large_var_cloned
#[no_mangle]
pub fn local_large_var_cloned(f: fn(Gigastruct)) {
    f(Gigastruct { does: 0, not: 1, have: 2, array: 3, members: 4 });

    // A new instance of `Gigastruct` is passed to `f()`, without any apparent
    // connection to this stack frame. Still, since instances of `Gigastruct`
    // are sufficiently large, it is allocated in the caller stack frame and
    // passed as a pointer. As such, this function is *also* protected by the
    // `strong` heuristic, just like `local_large_var_moved`. This is also the
    // case for pass-by-value of sufficiently large structs in C++:
    // ```
    // cat <<EOF | clang++ -O2 -fstack-protector-strong -S -x c++ - -o - | grep stack_chk
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
    // basic-NOT: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}


extern "C" {
    // A call to an external `alloca` function is *not* recognized as an
    // `alloca(3)` operation. This function is a compiler built-in, as the
    // man page explains. Clang translates it to an LLVM `alloca`
    // instruction with a count argument, which is also what the LLVM stack
    // protector heuristics looks for. The man page for `alloca(3)` details
    // a way to avoid using the compiler built-in: pass a -std=c11
    // argument, *and* don't include <alloca.h>. Though this leads to an
    // external alloca() function being called, it doesn't lead to stack
    // protection being included. It even fails with a linker error
    // "undefined reference to `alloca'". Example:
    // ```
    // cat<<EOF | clang -fstack-protector-strong -x c -std=c11 - -o /dev/null
    // #include <stdlib.h>
    // void * alloca(size_t);
    // void f(void (*g)(void*)) {
    //     void * p = alloca(10);
    //     g(p);
    // }
    // int main() { return 0; }
    // EOF
    // ```
    // The following tests demonstrate that calls to an external `alloca`
    // function in Rust also doesn't trigger stack protection.

    fn alloca(size: usize) -> *mut ();
}

// CHECK-LABEL: alloca_small_compile_time_constant_arg
#[no_mangle]
pub fn alloca_small_compile_time_constant_arg(f: fn(*mut ())) {
    f(unsafe { alloca(8) });

    // all: __stack_chk_fail
    // strong-NOT: __stack_chk_fail
    // basic-NOT: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

// CHECK-LABEL: alloca_large_compile_time_constant_arg
#[no_mangle]
pub fn alloca_large_compile_time_constant_arg(f: fn(*mut ())) {
    f(unsafe { alloca(9) });

    // all: __stack_chk_fail
    // strong-NOT: __stack_chk_fail
    // basic-NOT: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}


// CHECK-LABEL: alloca_dynamic_arg
#[no_mangle]
pub fn alloca_dynamic_arg(f: fn(*mut ()), n: usize) {
    f(unsafe { alloca(n) });

    // all: __stack_chk_fail
    // strong-NOT: __stack_chk_fail
    // basic-NOT: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

// The question then is: in what ways can Rust code generate array-`alloca`
// LLVM instructions? This appears to only be generated by
// rustc_codegen_ssa::traits::Builder::array_alloca() through
// rustc_codegen_ssa::mir::operand::OperandValue::store_unsized(). FWICT
// this is support for the "unsized locals" unstable feature:
// https://doc.rust-lang.org/unstable-book/language-features/unsized-locals.html.


// CHECK-LABEL: unsized_fn_param
#[no_mangle]
pub fn unsized_fn_param(s: [u8], l: bool, f: fn([u8])) {
    let n = if l { 1 } else { 2 };
    f(*Box::<[u8]>::from(&s[0..n])); // slice-copy with Box::from

    // Even though slices are conceptually passed by-value both into this
    // function and into `f()`, this is implemented with pass-by-reference
    // using a suitably constructed fat-pointer (as if the functions
    // accepted &[u8]). This function therefore doesn't need dynamic array
    // alloca, and is therefore not protected by the `strong` or `basic`
    // heuristics.


    // all: __stack_chk_fail
    // strong: __stack_chk_fail
    // basic-NOT: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}

// CHECK-LABEL: unsized_local
#[no_mangle]
pub fn unsized_local(s: &[u8], l: bool, f: fn(&mut [u8])) {
    let n = if l { 1 } else { 2 };
    let mut a: [u8] = *Box::<[u8]>::from(&s[0..n]); // slice-copy with Box::from
    f(&mut a);

    // This function allocates a slice as a local variable in its stack
    // frame. Since the size is not a compile-time constant, an array
    // alloca is required, and the function is protected by both the
    // `strong` and `basic` heuristic.

    // all: __stack_chk_fail
    // strong: __stack_chk_fail
    // basic: __stack_chk_fail
    // none-NOT: __stack_chk_fail
    // missing-NOT: __stack_chk_fail
}
