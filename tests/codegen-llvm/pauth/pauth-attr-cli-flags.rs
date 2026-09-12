// ignore-tidy-file-linelength
//@ only-pauthtest
//@ revisions: DEFAULT ALL DISABLE_JUMP DISABLE_AUTH_TRAPS DISABLE_CALLS DISABLE_INDIRCT_GOTOS DISABLE_RETURNS DISABLE_INTRINSICS DISABLE_TYPEINFO DISABLE_VT_PTR_ADDR DISABLE_VT_PTR_TYPE NONE

//@ add-minicore
#![crate_type = "lib"]
#![no_std]
#![no_core]
#![feature(no_core)]

extern crate minicore;

//@[DEFAULT] needs-llvm-components: aarch64
//@[DEFAULT] compile-flags: --target=aarch64-unknown-linux-pauthtest
//@[ALL] needs-llvm-components: aarch64
//@[ALL] compile-flags: --target=aarch64-unknown-linux-pauthtest -Zpointer-authentication=+aarch64-jump-table-hardening,+auth-traps,+calls,+indirect-gotos,+return-addresses
//@[DISABLE_JUMP] needs-llvm-components: aarch64
//@[DISABLE_JUMP] compile-flags: --target=aarch64-unknown-linux-pauthtest -Zpointer-authentication=-aarch64-jump-table-hardening
//@[DISABLE_AUTH_TRAPS] needs-llvm-components: aarch64
//@[DISABLE_AUTH_TRAPS] compile-flags: --target=aarch64-unknown-linux-pauthtest -Zpointer-authentication=-auth-traps
//@[DISABLE_CALLS] needs-llvm-components: aarch64
//@[DISABLE_CALLS] compile-flags: --target=aarch64-unknown-linux-pauthtest -Zpointer-authentication=-calls
//@[DISABLE_INDIRCT_GOTOS] needs-llvm-components: aarch64
//@[DISABLE_INDIRCT_GOTOS] compile-flags: --target=aarch64-unknown-linux-pauthtest -Zpointer-authentication=-indirect-gotos
//@[DISABLE_RETURNS] needs-llvm-components: aarch64
//@[DISABLE_RETURNS] compile-flags: --target=aarch64-unknown-linux-pauthtest -Zpointer-authentication=-return-addresses
//@[DISABLE_INTRINSICS] needs-llvm-components: aarch64
//@[DISABLE_INTRINSICS] compile-flags: --target=aarch64-unknown-linux-pauthtest -Zpointer-authentication=-intrinsics
//@[DISABLE_TYPEINFO] needs-llvm-components: aarch64
//@[DISABLE_TYPEINFO] compile-flags: --target=aarch64-unknown-linux-pauthtest -Zpointer-authentication=-typeinfo-vt-ptr-discrimination
//@[DISABLE_VT_PTR_ADDR] needs-llvm-components: aarch64
//@[DISABLE_VT_PTR_ADDR] compile-flags: --target=aarch64-unknown-linux-pauthtest -Zpointer-authentication=-vt-ptr-addr-discrimination
//@[DISABLE_VT_PTR_TYPE] needs-llvm-components: aarch64
//@[DISABLE_VT_PTR_TYPE] compile-flags: --target=aarch64-unknown-linux-pauthtest -Zpointer-authentication=-vt-ptr-type-discrimination
//@[NONE] needs-llvm-components: aarch64
//@[NONE] compile-flags: --target=aarch64-unknown-linux-pauthtest  -Zpointer-authentication=-aarch64-jump-table-hardening,-auth-traps,-calls,-indirect-gotos,-return-addresses,-init-fini,-init-fini-address-discrimination,-intrinsics,-typeinfo-vt-ptr-discrimination,-vt-ptr-addr-discrimination,-vt-ptr-type-discrimination

// CHECK: define {{.*}} @{{.*}}main{{.*}} [[ATTR_MAIN:#[0-9]+]]
#[inline(never)]
pub fn main() {}
// DEFAULT: attributes [[ATTR_MAIN]] = { {{.*}}"aarch64-jump-table-hardening"
// DEFAULT-SAME: "ptrauth-auth-traps"
// DEFAULT-SAME: "ptrauth-calls"
// DEFAULT-SAME: "ptrauth-indirect-gotos"
// DEFAULT-SAME: "ptrauth-returns"
// DEFAULT: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// DEFAULT-NEXT: !{i32 1, !"aarch64-elf-pauthabi-version", i32 1791}

// DISABLE_JUMP-NOT: aarch64-jump-table-hardening
// DISABLE_JUMP: attributes [[ATTR_MAIN]] = { {{.*}}"ptrauth-auth-traps"
// DISABLE_JUMP-SAME: "ptrauth-calls"
// DISABLE_JUMP-SAME: "ptrauth-indirect-gotos"
// DISABLE_JUMP-SAME: "ptrauth-returns"
// DISABLE_JUMP: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// DISABLE_JUMP-NEXT: !{i32 1, !"aarch64-elf-pauthabi-version", i32 1791}

// DISABLE_AUTH_TRAPS-NOT: ptrauth-auth-traps
// DISABLE_AUTH_TRAPS: attributes [[ATTR_MAIN]] = { {{.*}}"aarch64-jump-table-hardening"
// DISABLE_AUTH_TRAPS-SAME: "ptrauth-calls"
// DISABLE_AUTH_TRAPS-SAME: "ptrauth-indirect-gotos"
// DISABLE_AUTH_TRAPS-SAME: "ptrauth-returns"
// DISABLE_AUTH_TRAPS: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// DISABLE_AUTH_TRAPS-NEXT !{i32 1, !"aarch64-elf-pauthabi-version", i32 1783}

// DISABLE_CALLS-NOT: ptrauth-calls
// DISABLE_CALLS: attributes [[ATTR_MAIN]] = { {{.*}}"aarch64-jump-table-hardening"
// DISABLE_CALLS-SAME: "ptrauth-auth-traps"
// DISABLE_CALLS-SAME: "ptrauth-indirect-gotos"
// DISABLE_CALLS-SAME: "ptrauth-returns"
// DISABLE_CALLS: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// DISABLE_CALLS-SAME-NEXT: !{i32 1, !"aarch64-elf-pauthabi-version", i32 1789}

// DISABLE_INDIRCT_GOTOS-NOT: ptrauth-indirect-gotos
// DISABLE_INDIRCT_GOTOS: attributes [[ATTR_MAIN]] = { {{.*}}"aarch64-jump-table-hardening"
// DISABLE_INDIRCT_GOTOS-SAME: "ptrauth-auth-traps"
// DISABLE_INDIRCT_GOTOS-SAME: "ptrauth-calls"
// DISABLE_INDIRCT_GOTOS-SAME: "ptrauth-returns"
// DISABLE_INDIRCT_GOTOS: !{i32 1, !"aarch64-elf-pauthabi-version", i32 1279}
// DISABLE_INDIRCT_GOTOS-NEXT: !{i32 1, !"ptrauth-sign-personality", i32 1}

// DISABLE_RETURNS-NOT: ptrauth-returns
// DISABLE_RETURNS: attributes [[ATTR_MAIN]] = { {{.*}}"aarch64-jump-table-hardening"
// DISABLE_RETURNS-SAME: "ptrauth-auth-traps"
// DISABLE_RETURNS-SAME: "ptrauth-calls"
// DISABLE_RETURNS-SAME: "ptrauth-indirect-gotos"
// DISABLE_RETURNS: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// DISABLE_RETURNS-NEXT: !{i32 1, !"aarch64-elf-pauthabi-version", i32 1787}

// DISABLE_INTRINSICS: attributes [[ATTR_MAIN]] = { {{.*}}"aarch64-jump-table-hardening"
// DISABLE_INTRINSICS-SAME: "ptrauth-auth-traps"
// DISABLE_INTRINSICS-SAME: "ptrauth-calls"
// DISABLE_INTRINSICS-SAME: "ptrauth-indirect-gotos"
// DISABLE_INTRINSICS-SAME: "ptrauth-returns"
// DISABLE_INTRINSICS: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// DISABLE_INTRINSICS-NEXT: !{i32 1, !"aarch64-elf-pauthabi-version", i32 1790}

// DISABLE_TYPEINFO: attributes [[ATTR_MAIN]] = { {{.*}}"aarch64-jump-table-hardening"
// DISABLE_TYPEINFO-SAME: "ptrauth-auth-traps"
// DISABLE_TYPEINFO-SAME: "ptrauth-calls"
// DISABLE_TYPEINFO-SAME: "ptrauth-indirect-gotos"
// DISABLE_TYPEINFO-SAME: "ptrauth-returns"
// DISABLE_TYPEINFO: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// DISABLE_TYPEINFO-NEXT: !{i32 1, !"aarch64-elf-pauthabi-version", i32 767}

// DISABLE_VT_PTR_ADDR: attributes [[ATTR_MAIN]] = { {{.*}}"aarch64-jump-table-hardening"
// DISABLE_VT_PTR_ADDR-SAME: "ptrauth-auth-traps"
// DISABLE_VT_PTR_ADDR-SAME: "ptrauth-calls"
// DISABLE_VT_PTR_ADDR-SAME: "ptrauth-indirect-gotos"
// DISABLE_VT_PTR_ADDR-SAME: "ptrauth-returns"
// DISABLE_VT_PTR_ADDR: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// DISABLE_VT_PTR_ADDR-NEXT: !{i32 1, !"aarch64-elf-pauthabi-version", i32 1775}

// DISABLE_VT_PTR_TYPE: attributes [[ATTR_MAIN]] = { {{.*}}"aarch64-jump-table-hardening"
// DISABLE_VT_PTR_TYPE-SAME: "ptrauth-auth-traps"
// DISABLE_VT_PTR_TYPE-SAME: "ptrauth-calls"
// DISABLE_VT_PTR_TYPE-SAME: "ptrauth-indirect-gotos"
// DISABLE_VT_PTR_TYPE-SAME: "ptrauth-returns"
// DISABLE_VT_PTR_TYPE: !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
// DISABLE_VT_PTR_TYPE-NEXT: !{i32 1, !"aarch64-elf-pauthabi-version", i32 1759}

// NONE-NOT: ptrauth-returns
// NONE-NOT: aarch64-jump-table-hardening
// NONE-NOT: ptrauth-auth-traps
// NONE-NOT: ptrauth-calls
// NONE-NOT: ptrauth-indirect-gotos
// NONE-NOT: aarch64-elf-pauthabi-platform
// NONE-NOT: aarch64-elf-pauthabi-version
