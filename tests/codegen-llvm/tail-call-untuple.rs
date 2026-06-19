//! Regression test for issue <https://github.com/rust-lang/rust/issues/158017>.
// Checks that we pass the caller's argument slots as arguments at tail call,
// because the caller's stack frame is overwritten by the callee's stack frame.
// In LLVM, Calls marked 'tail' cannot read or write allocas from the current frame
// because the current frame might be destroyed by the time they run. These writes will be
// eliminated by DSE.
//@ add-minicore
//@ revisions: x64-linux i686-linux i686-windows
//@ compile-flags: -C opt-level=3
//@[x64-linux] compile-flags: --target x86_64-unknown-linux-gnu
//@[x64-linux] needs-llvm-components: x86
//@[i686-linux] compile-flags: --target i686-unknown-linux-gnu
//@[i686-linux] needs-llvm-components: x86
//@[i686-windows] compile-flags: --target i686-pc-windows-msvc
//@[i686-windows] needs-llvm-components: x86

#![crate_type = "lib"]
#![feature(explicit_tail_calls, no_core, unboxed_closures)]
#![expect(incomplete_features)]
#![no_std]
#![no_core]

extern crate minicore;

struct Indirect(u64, u64, u64, u64);

// CHECK-LABEL: @caller_untuple_1
// CHECK-SAME: (ptr {{.*}}[[A:%.*]])
// CHECK-NEXT: start:
// CHECK-NEXT: musttail call {{.*}}i64 @callee_untuple_1(ptr {{.*}}[[A]])
#[unsafe(no_mangle)]
extern "rust-call" fn caller_untuple_1((a,): (Indirect,)) -> u64 {
    become callee_untuple_1((a,));
}

// CHECK-LABEL: @caller_untuple_1_const
// CHECK-SAME: (ptr {{.*}}[[A:%.*]])
// x64-linux: store i64 1, ptr [[A]]
// i686-linux: store <2 x i64> <i64 1, i64 2>, ptr [[A]]
// i686-windows: store <2 x i64> <i64 1, i64 2>, ptr [[A]]
// CHECK: musttail call {{.*}}i64 @callee_untuple_1(ptr {{.*}}[[A]])
#[unsafe(no_mangle)]
extern "rust-call" fn caller_untuple_1_const((_,): (Indirect,)) -> u64 {
    become callee_untuple_1((Indirect(1, 2, 3, 4),));
}

// CHECK-LABEL: @caller_untuple_2
// CHECK-SAME: (ptr {{.*}}[[A:%.*]], ptr {{.*}}[[B:%.*]])
// CHECK-NEXT: start:
// CHECK-NEXT: musttail call {{.*}}i64 @callee_untuple_2(ptr {{.*}}[[A]], ptr {{.*}}[[B]])
#[unsafe(no_mangle)]
extern "rust-call" fn caller_untuple_2((a, b): (Indirect, Indirect)) -> u64 {
    become callee_untuple_2((a, b));
}

// CHECK-LABEL: @caller_untuple_2_swapper
// CHECK-SAME: (ptr {{.*}}[[A:%.*]], ptr {{.*}}[[B:%.*]])
// CHECK: call void @llvm.memcpy.{{.+}}(ptr {{.*}}[[TMP:%.*]], ptr {{.*}}[[A]]
// CHECK: call void @llvm.memcpy.{{.+}}(ptr {{.*}}[[A]], ptr {{.*}}[[B]]
// CHECK: call void @llvm.memcpy.{{.+}}(ptr {{.*}}[[B]], ptr {{.*}}[[TMP]]
// CHECK: musttail call {{.*}}i64 @callee_untuple_2(ptr {{.*}}[[A]], ptr {{.*}}[[B]])
#[unsafe(no_mangle)]
extern "rust-call" fn caller_untuple_2_swapper((a, b): (Indirect, Indirect)) -> u64 {
    become callee_untuple_2((b, a));
}

mod swap_self {
    type Tuple = (u64, u64, u64, u64);

    trait X {
        extern "rust-call" fn swap_self(self, args: Tuple) -> u8;
        extern "rust-call" fn swap_self_helper(self, args: Tuple) -> u8;
    }

    impl X for Tuple {
        // CHECK-LABEL: @swap_self
        // CHECK-SAME: (ptr {{.*}}[[SELF:%.*]], i64 {{.*}}[[TUPLE_0:%.*]], i64 {{.*}}[[TUPLE_1:%.*]], i64 {{.*}}[[TUPLE_2:%.*]], i64 {{.*}}[[TUPLE_3:%.*]])
        // CHECK: [[SELF_0:%.*]] = load i64, ptr [[SELF]]
        // CHECK: [[PTR_1:%.*]] = getelementptr inbounds {{.*}}i8, ptr %self, {{i32|i64}} 8
        // CHECK: [[SELF_1:%.*]] = load i64, ptr [[PTR_1]]
        // CHECK: [[PTR_2:%.*]] = getelementptr inbounds {{.*}}i8, ptr %self, {{i32|i64}} 16
        // CHECK: [[SELF_2:%.*]] = load i64, ptr [[PTR_2]]
        // CHECK: [[PTR_3:%.*]] = getelementptr inbounds {{.*}}i8, ptr %self, {{i32|i64}} 24
        // CHECK: [[SELF_3:%.*]] = load i64, ptr [[PTR_3]]
        // CHECK: store i64 [[TUPLE_0]], ptr [[SELF]]
        // CHECK: store i64 [[TUPLE_1]], ptr [[PTR_1]]
        // CHECK: store i64 [[TUPLE_2]], ptr [[PTR_2]]
        // CHECK: store i64 [[TUPLE_3]], ptr [[PTR_3]]
        // CHECK: musttail call {{.*}}i8 @swap_self_helper(ptr {{.*}}[[SELF]], i64 {{.*}}[[SELF_0]], i64 {{.*}}[[SELF_1]], i64 {{.*}}[[SELF_2]], i64 {{.*}}[[SELF_3]])
        #[unsafe(no_mangle)]
        extern "rust-call" fn swap_self(self, args: Tuple) -> u8 {
            become args.swap_self_helper(self)
        }

        #[inline(never)]
        #[unsafe(no_mangle)]
        extern "rust-call" fn swap_self_helper(self, args: Tuple) -> u8 {
            opaque_tuple(self);
            opaque_tuple(args);
            0
        }
    }

    unsafe extern "Rust" {
        safe fn opaque_tuple(_: Tuple);
    }
}

unsafe extern "Rust" {
    safe fn opaque(_: u64);
}

#[inline(never)]
#[unsafe(no_mangle)]
extern "rust-call" fn callee_untuple_1((a,): (Indirect,)) -> u64 {
    opaque(a.0);
    a.0
}

#[inline(never)]
#[unsafe(no_mangle)]
extern "rust-call" fn callee_untuple_2((a, b): (Indirect, Indirect)) -> u64 {
    opaque(a.0);
    opaque(b.0);
    a.0
}
