# `verbose-asm`

The tracking issue for this feature is: [#126802](https://github.com/rust-lang/rust/issues/126802).

------------------------

This enables passing `-Zverbose-asm` to get contextual comments added by LLVM.

Sample code:

```rust
#[no_mangle]
pub fn foo(a: i32, b: i32) -> i32 {
    a + b
}
```

Default output:

```asm
foo:
        push    rax
        add     edi, esi
        mov     dword ptr [rsp + 4], edi
        seto    al
        jo      .LBB0_2
        mov     eax, dword ptr [rsp + 4]
        pop     rcx
        ret
.LBB0_2:
        lea     rdi, [rip + .L__unnamed_1]
        mov     rax, qword ptr [rip + core::panicking::panic_const::panic_const_add_overflow::h9c85248fe0d735b2@GOTPCREL]
        call    rax

.L__unnamed_2:
        .ascii  "/app/example.rs"

.L__unnamed_1:
        .quad   .L__unnamed_2
        .asciz  "\017\000\000\000\000\000\000\000\004\000\000\000\005\000\000"
```

With `-Zverbose-asm`:

```asm
foo:                                    # @foo
# %bb.0:
        push    rax
        add     edi, esi
        mov     dword ptr [rsp + 4], edi        # 4-byte Spill
        seto    al
        jo      .LBB0_2
# %bb.1:
        mov     eax, dword ptr [rsp + 4]        # 4-byte Reload
        pop     rcx
        ret
.LBB0_2:
        lea     rdi, [rip + .L__unnamed_1]
        mov     rax, qword ptr [rip + core::panicking::panic_const::panic_const_add_overflow::h9c85248fe0d735b2@GOTPCREL]
        call    rax
                                        # -- End function
.L__unnamed_2:
        .ascii  "/app/example.rs"

.L__unnamed_1:
        .quad   .L__unnamed_2
        .asciz  "\017\000\000\000\000\000\000\000\004\000\000\000\005\000\000"

                                        # DW_AT_external
```
