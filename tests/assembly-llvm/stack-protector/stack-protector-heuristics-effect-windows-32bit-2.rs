//@ revisions: all strong none missing
//@ assembly-output: emit-asm
//@ only-windows
//@ only-msvc
//@ ignore-64bit 64-bit table based SEH has slightly different behaviors than classic SEH
//@ [all] compile-flags: -Z stack-protector=all
//@ [strong] compile-flags: -Z stack-protector=strong
//@ [none] compile-flags: -Z stack-protector=none
//@ compile-flags: -C opt-level=2 -Z merge-functions=disabled

#![crate_type = "lib"]
#![allow(internal_features)]
#![feature(unsized_fn_params)]

extern "C" {
    fn strcpy(dest: *mut u8, src: *const u8) -> *mut u8;
    fn printf(fmt: *const u8, ...) -> i32;
    fn funcall(p: *mut i32);
    fn funcall2(p: *mut *mut i32);
    fn funfloat(p: *mut f64);
    fn funfloat2(p: *mut *mut f64);
    fn testi_aux() -> f64;
    fn getp() -> *mut i32;
    fn dummy(_: ...) -> i32;

    static STR: [u8; 1];
}

extern "C-unwind" {
    fn except(p: *mut i32);
}

#[repr(C)]
struct Pair {
    a: i32,
    b: i32,
}

#[repr(C)]
struct Nest {
    first: Pair,
    second: Pair,
}

// test1: array of [16 x i8]
// CHECK-LABEL: test1{{:|\[}}
#[no_mangle]
pub fn test1(a: *const u8) {
    let mut buf: [u8; 16] = [0; 16];

    unsafe {
        strcpy(buf.as_mut_ptr(), a);
        printf(STR.as_ptr(), buf.as_ptr());
    }

    // all: __security_check_cookie
    // strong: __security_check_cookie
    // none-NOT: __security_check_cookie
    // missing-NOT: __security_check_cookie
}

// test2: array [4 x i8]
// CHECK-LABEL: test2{{:|\[}}
#[no_mangle]
pub fn test2(a: *const u8) {
    let mut buf: [u8; 4] = [0; 4];

    unsafe {
        strcpy(buf.as_mut_ptr(), a);
        printf(STR.as_ptr(), buf.as_ptr());
    }

    // all: __security_check_cookie
    // strong: __security_check_cookie
    // none-NOT: __security_check_cookie
    // missing-NOT: __security_check_cookie
}

// test3: no arrays / no nested arrays
// CHECK-LABEL: test3{{:|\[}}
#[no_mangle]
pub fn test3(a: *const u8) {
    unsafe {
        printf(STR.as_ptr(), a);
    }

    // all: __security_check_cookie
    // strong-NOT: __security_check_cookie
    // none-NOT: __security_check_cookie
    // missing-NOT: __security_check_cookie
}

// test4: Address-of local taken (j = &a)
// CHECK-LABEL: test4{{:|\[}}
#[no_mangle]
pub unsafe extern "C" fn test4() {
    let mut a: i32 = 0;

    let mut j: *mut i32 = std::ptr::null_mut();

    let tmp = std::ptr::read_volatile(&a);
    let tmp2 = tmp.wrapping_add(1);
    std::ptr::write_volatile(&mut a, tmp2);

    std::ptr::write_volatile(&mut j, &mut a);

    // all: __security_check_cookie
    // strong: __security_check_cookie
    // none-NOT: __security_check_cookie
    // missing-NOT: __security_check_cookie
}

// test5: PtrToInt Cast
// CHECK-LABEL: test5{{:|\[}}
#[no_mangle]
pub fn test5(a: i32) {
    let ptr_val: usize = &a as *const i32 as usize;

    unsafe {
        printf(STR.as_ptr(), ptr_val);
    }

    // all: __security_check_cookie
    // strong: __security_check_cookie
    // none-NOT: __security_check_cookie
    // missing-NOT: __security_check_cookie
}

// test6: Passing addr-of to function call
// CHECK-LABEL: test6{{:|\[}}
#[no_mangle]
pub fn test6(mut b: i32) {
    unsafe {
        funcall(&mut b as *mut i32);
    }

    // all: __security_check_cookie
    // strong: __security_check_cookie
    // none-NOT: __security_check_cookie
    // missing-NOT: __security_check_cookie
}

// test7: Addr-of in select instruction
// CHECK-LABEL: test7{{:|\[}}
#[no_mangle]
pub fn test7() {
    let x: f64;

    unsafe {
        let call = testi_aux();
        x = call;

        let y: *const f64 = if call > 0.0 { &x as *const f64 } else { std::ptr::null() };

        printf(STR.as_ptr(), y);
    }

    // all: __security_check_cookie
    // strong: __security_check_cookie
    // none-NOT: __security_check_cookie
    // missing-NOT: __security_check_cookie
}

// test8: Addr-of in phi instruction
// CHECK-LABEL: test8{{:|\[}}
#[no_mangle]
pub fn test8() {
    let mut _x: f64;

    unsafe {
        let call = testi_aux();
        _x = call;

        let y: *const f64;

        if call > 3.14 {
            let call1 = testi_aux();
            _x = call1;
            y = std::ptr::null();
        } else {
            if call > 1.0 {
                y = &_x;
            } else {
                y = std::ptr::null();
            }
        }

        printf(STR.as_ptr(), y);
    }

    // all: __security_check_cookie
    // strong: __security_check_cookie
    // none-NOT: __security_check_cookie
    // missing-NOT: __security_check_cookie
}

// test9: Addr-of struct element(GEP followed by store)
// CHECK-LABEL: test9{{:|\[}}
#[no_mangle]
pub fn test9() {
    let mut c = Pair { a: 0, b: 0 };
    let b: *mut i32;

    unsafe {
        let y: *mut i32 = &mut c.b;

        b = y;

        printf(STR.as_ptr(), b);
    }

    // all: __security_check_cookie
    // strong: __security_check_cookie
    // none-NOT: __security_check_cookie
    // missing-NOT: __security_check_cookie
}

// test10: Addr-of struct element, GEP followed by ptrtoint
// CHECK-LABEL: test10{{:|\[}}
#[no_mangle]
pub fn test10() {
    let mut c = Pair { a: 0, b: 0 };

    unsafe {
        let y: *mut i32 = &mut c.b;

        let addr: i64 = y as i64;

        printf(STR.as_ptr(), addr);
    }

    // all: __security_check_cookie
    // strong: __security_check_cookie
    // none-NOT: __security_check_cookie
    // missing-NOT: __security_check_cookie
}

// test11: Addr-of struct element, GEP followed by callinst
// CHECK-LABEL: test11{{:|\[}}
#[no_mangle]
pub fn test11() {
    let mut c = Pair { a: 0, b: 0 };

    unsafe {
        let y: *mut i32 = &mut c.b;

        printf(STR.as_ptr(), y);
    }

    // all: __security_check_cookie
    // strong: __security_check_cookie
    // none-NOT: __security_check_cookie
    // missing-NOT: __security_check_cookie
}

// test12: Addr-of a local, optimized into a GEP (e.g., &a - 12)
// CHECK-LABEL: test12{{:|\[}}
#[no_mangle]
pub fn test12() {
    let mut a: i32 = 0;

    unsafe {
        let add_ptr = (&mut a as *mut i32).offset(-12);

        printf(STR.as_ptr(), add_ptr);
    }

    // all: __security_check_cookie
    // strong: __security_check_cookie
    // none-NOT: __security_check_cookie
    // missing-NOT: __security_check_cookie
}

// test13: Addr-of a local cast to a ptr of a different type
// (e.g., int a; ... ; ptr b = &a;)
// CHECK-LABEL: test13{{:|\[}}
#[no_mangle]
pub fn test13() {
    let mut a: i32 = 0;

    unsafe {
        let mut b: *mut i32 = std::ptr::null_mut();

        std::ptr::write_volatile(&mut b, &mut a); // avoid ptr b from optimization
        let tmp = std::ptr::read_volatile(&b);

        printf(STR.as_ptr(), tmp);
    }

    // all: __security_check_cookie
    // strong: __security_check_cookie
    // none-NOT: __security_check_cookie
    // missing-NOT: __security_check_cookie
}

// test14: Addr-of a local cast to a ptr of a different type (optimized)
// (e.g., int a; ... ; ptr b = &a;)
// CHECK-LABEL: test14{{:|\[}}
#[no_mangle]
pub fn test14() {
    let a: i32 = 0;
    unsafe {
        funfloat((&a as *const i32).cast::<f64>() as *mut f64);
    }

    // all: __security_check_cookie
    // strong: __security_check_cookie
    // none-NOT: __security_check_cookie
    // missing-NOT: __security_check_cookie
}

// test15: Addr-of a variable passed into an invoke instruction
// CHECK-LABEL: test15{{:|\[}}
#[no_mangle]
pub unsafe extern "C" fn test15() -> i32 {
    let mut a: i32 = 0;

    except(&mut a as *mut i32);

    0

    // stack protector does not generated by LLVM because of Windows SEH.

    // all-NOT: __security_check_cookie
    // strong-NOT: __security_check_cookie
    // none-NOT: __security_check_cookie
    // missing-NOT: __security_check_cookie
}

// test16: Addr-of a struct element passed into an invoke instruction
// (GEP followed by an invoke)
// CHECK-LABEL: test16{{:|\[}}
#[no_mangle]
pub unsafe extern "C" fn test16() -> i32 {
    let mut c = Pair { a: 0, b: 0 };

    except(&mut c.a as *mut i32);

    0

    // stack protector does not generated by LLVM because of Windows SEH.

    // all-NOT: __security_check_cookie
    // strong-NOT: __security_check_cookie
    // none-NOT: __security_check_cookie
    // missing-NOT: __security_check_cookie
}

// test17: Addr-of a pointer
// CHECK-LABEL: test17{{:|\[}}
#[no_mangle]
pub unsafe extern "C" fn test17() {
    let mut a: *mut i32 = getp();

    let mut _b: *mut *mut i32 = std::ptr::null_mut();

    std::ptr::write_volatile(&mut _b, &mut a);

    let tmp = std::ptr::read_volatile(&_b);

    funcall2(tmp);

    // all: __security_check_cookie
    // strong: __security_check_cookie
    // none-NOT: __security_check_cookie
    // missing-NOT: __security_check_cookie
}

// test18: Addr-of a casted pointer
// CHECK-LABEL: test18{{:|\[}}
#[no_mangle]
pub unsafe extern "C" fn test18() {
    let mut a: *mut i32 = getp();

    let mut _b: *mut *mut i32 = std::ptr::null_mut();

    std::ptr::write_volatile(&mut _b, &mut a);

    let tmp = std::ptr::read_volatile(&_b);

    funfloat2(tmp as *mut *mut f64);

    // all: __security_check_cookie
    // strong: __security_check_cookie
    // none-NOT: __security_check_cookie
    // missing-NOT: __security_check_cookie
}

// test19: array of [4 x i32]
// CHECK-LABEL: test19{{:|\[}}
#[no_mangle]
pub unsafe extern "C" fn test19() -> i32 {
    let a: [i32; 4] = [0; 4];

    let _whole = std::ptr::read_volatile(&a as *const _); // avoid array a from optimization

    std::ptr::read_volatile(&a[0])

    // all: __security_check_cookie
    // strong: __security_check_cookie
    // none-NOT: __security_check_cookie
    // missing-NOT: __security_check_cookie
}

// test20: Nested structure, no arrays, no address-of expressions
// Verify that the resulting gep-of-gep does not incorrectly trigger
// a stack protector
// CHECK-LABEL: test20{{:|\[}}
#[no_mangle]
pub unsafe extern "C" fn test20() {
    let c = Nest { first: Pair { a: 10, b: 11 }, second: Pair { a: 20, b: 21 } };

    let whole: Nest = std::ptr::read_volatile(&c);

    let v: i32 = whole.second.a;

    printf(STR.as_ptr(), v);

    // strong-NOT: __security_check_cookie
}

// test21: Address-of a structure taken in a function with a loop where
// the alloca is an incoming value to a PHI node and a use of that PHI
// node is also an incoming value
// Verify that the address-of analysis does not get stuck in infinite
// recursion when chasing the alloca through the PHI nodes
// CHECK-LABEL: test21{{:|\[}}
#[no_mangle]
pub unsafe extern "C" fn test21() -> i32 {
    let mut tmp: *mut u8 = std::ptr::null_mut();
    let tmp_ptr: *mut *mut u8 = &mut tmp;

    let tmp1 = dummy(tmp_ptr);

    let cur = std::ptr::read_volatile(tmp_ptr);

    let v = (cur as usize as i64) as i32;

    if v > 0 {
        let mut phi_ptr: *mut u8 = cur;
        let mut phi_idx: i64 = 1;
        let mut phi_acc: i32 = tmp1;

        loop {
            let b = std::ptr::read_volatile(phi_ptr as *const u8);
            let cond = b == 1u8;
            let plus = phi_acc.wrapping_add(8);
            let next_acc = if cond { plus } else { phi_acc };

            if (phi_idx as i32) == v {
                dummy(next_acc);
                break;
            }

            let slot = tmp_ptr.add(phi_idx as usize);
            let next = std::ptr::read_volatile(slot);
            phi_ptr = next;
            phi_idx += 1;
            phi_acc = next_acc;
        }
    } else {
        dummy(tmp1);
    }

    0

    // strong: __security_check_cookie
}

// CHECK-LABEL: IgnoreIntrinsicTest{{:|\[}}
#[no_mangle]
pub unsafe extern "C" fn IgnoreIntrinsicTest() -> i32 {
    let mut x: i32 = 0;

    std::ptr::write_volatile(&mut x, 1);

    let y = std::ptr::read_volatile(&x);

    let result = y.wrapping_mul(42);

    result

    // strong-NOT: __security_check_cookie
}
