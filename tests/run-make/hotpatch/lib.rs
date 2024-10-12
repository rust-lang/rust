// hotpatch has two requirements:
// 1) the first instruction of a functin must be at least two bytes long
// 2) there must not be a jump to the first instruction

// The LLVM attribute we use '"patchable-function", "prologue-short-redirect"' only ensures 1)
// However in practice 2) rarely matters. Its rare that it occurs and the problems it caused can be
// avoided by the hotpatch tool.
// In this test we check if 1) is ensured by inserted nops as needed

// ----------------------------------------------------------------------------------------------

// empty_fn just returns. Note that 'ret' is a single byte instruction, but hotpatch requires
// a two or more byte instructions to be at the start of the functions.
// Preferably we would also tests a different single byte instruction,
// but I was not able to find an example with another one byte intstruction.

// check that if the first instruction is just a single byte, so our test is valid
// CHECK-LABEL: <empty_fn>:
// CHECK-NOT: 0: {{[0-9a-f][0-9a-f]}} {{[0-9a-f][0-9a-f]}} {{.*}}

// check that the first instruction is at least 2 bytes long
// HOTPATCH-LABEL: <empty_fn>:
// HOTPATCH-NEXT: 0: {{[0-9a-f][0-9a-f]}} {{[0-9a-f][0-9a-f]}} {{.*}}

#[no_mangle]
#[inline(never)]
pub fn empty_fn() {}
