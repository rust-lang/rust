// Test that the `cs` prefix is (not) added into a `call` and a `jmp` to the
// indirect thunk when the `-Zindirect-branch-cs-prefix` flag is (not) set.

//@ revisions: unset set
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3 -Cunsafe-allow-abi-mismatch=retpoline,retpoline-external-thunk,indirect-branch-cs-prefix -Zretpoline-external-thunk
//@ [set] compile-flags: -Zindirect-branch-cs-prefix
//@ only-x86_64
//@ ignore-apple Symbol is called `___x86_indirect_thunk` (Darwin's extra underscore)

#![crate_type = "lib"]

// CHECK-LABEL: foo:
#[no_mangle]
pub fn foo(g: fn()) {
    // unset-NOT: cs
    // unset: callq {{__x86_indirect_thunk.*}}
    // set: cs
    // set-NEXT: callq {{__x86_indirect_thunk.*}}
    g();

    // unset-NOT: cs
    // unset: jmp {{__x86_indirect_thunk.*}}
    // set: cs
    // set-NEXT: jmp {{__x86_indirect_thunk.*}}
    g();
}
