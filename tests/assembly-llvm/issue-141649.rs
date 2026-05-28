//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3

//@ revisions: aarch64
//@ [aarch64] only-aarch64

//@ revisions: linux-x86_64
//@ [linux-x86_64] compile-flags: --target x86_64-unknown-linux-gnu
//@ [linux-x86_64] only-x86_64-unknown-linux-gnu
//@ [linux-x86_64] needs-llvm-components: x86

//@ revisions: windows-x86_64-msvc
//@ [windows-x86_64-msvc] compile-flags: --target x86_64-pc-windows-msvc
//@ [windows-x86_64-msvc] needs-llvm-components: x86
//@ [windows-x86_64-msvc] only-x86_64-pc-windows-msvc

#![crate_type = "lib"]

// Non-overlapping scopes should reuse of the same stack allocation.

pub struct WithOffset<T> {
    pub data: T,
    pub offset: usize,
}

#[inline(never)]
pub fn peak_w(w: &WithOffset<&[u8; 16]>) {
    std::hint::black_box(w);
}

#[inline(never)]
pub fn use_w(w: WithOffset<&[u8; 16]>) {
    std::hint::black_box(w);
}

// CHECK-LABEL: scoped_two_small_structs
#[no_mangle]
pub fn scoped_two_small_structs(buf: [u8; 16]) {
    {
        let w = WithOffset { data: &buf, offset: 0 };

        peak_w(&w);
        use_w(w);
    }
    {
        let w2 = WithOffset { data: &buf, offset: 1 };

        peak_w(&w2);
        use_w(w2);
    }
    // linux-x86_64: subq $16, %rsp
    // windows-x86_64-msvc: subq $48, %rsp
    // aarch64: sub sp, sp, #48
}

// CHECK-LABEL: scoped_three_small_structs
#[no_mangle]
pub fn scoped_three_small_structs(buf: [u8; 16]) {
    {
        let w = WithOffset { data: &buf, offset: 0 };

        peak_w(&w);
        use_w(w);
    }
    {
        let w2 = WithOffset { data: &buf, offset: 1 };

        peak_w(&w2);
        use_w(w2);
    }
    {
        let w3 = WithOffset { data: &buf, offset: 1 };

        peak_w(&w3);
        use_w(w3);
    }
    // Should be the same stack usage as the two struct version.
    // linux-x86_64: subq $16, %rsp
    // windows-x86_64-msvc: subq $48, %rsp
    // aarch64: sub sp, sp, #48
}
