// ignore-tidy-linelength
//@ only-pauthtest
//@ revisions: DEFAULT ALL DISABLE_JUMP DISABLE_AUTH_TRAPS DISABLE_CALLS DISABLE_INDIRCT_GOTOS DISABLE_RETURNS DISABLE_INTRINSICS DISABLE_TYPEINFO DISABLE_VT_PTR_ADDR DISABLE_VT_PTR_TYPE NONE

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
//@[NONE] needs-llvm-components: aarch64
//@[NONE] compile-flags: --target=aarch64-unknown-linux-pauthtest  -Zpointer-authentication=-aarch64-jump-table-hardening,-auth-traps,-calls,-indirect-gotos,-return-addresses

// CHECK: define {{.*}} @main{{.*}} [[ATTR_MAIN:#[0-9]+]]
fn main() {}
// DEFAULT: attributes [[ATTR_MAIN]] = { {{.*}}"aarch64-jump-table-hardening"
// DEFAULT-SAME: "ptrauth-auth-traps"
// DEFAULT-SAME: "ptrauth-calls"
// DEFAULT-SAME: "ptrauth-indirect-gotos"
// DEFAULT-SAME: "ptrauth-returns"

// DISABLE_JUMP-NOT: aarch64-jump-table-hardening
// DISABLE_JUMP: attributes [[ATTR_MAIN]] = { {{.*}}"ptrauth-auth-traps"
// DISABLE_JUMP-SAME: "ptrauth-calls"
// DISABLE_JUMP-SAME: "ptrauth-indirect-gotos"
// DISABLE_JUMP-SAME: "ptrauth-returns"

// DISABLE_AUTH_TRAPS-NOT: ptrauth-auth-traps
// DISABLE_AUTH_TRAPS: attributes [[ATTR_MAIN]] = { {{.*}}"aarch64-jump-table-hardening"
// DISABLE_AUTH_TRAPS-SAME: "ptrauth-calls"
// DISABLE_AUTH_TRAPS-SAME: "ptrauth-indirect-gotos"
// DISABLE_AUTH_TRAPS-SAME: "ptrauth-returns"

// DISABLE_CALLS-NOT: ptrauth-calls
// DISABLE_CALLS: attributes [[ATTR_MAIN]] = { {{.*}}"aarch64-jump-table-hardening"
// DISABLE_CALLS-SAME: "ptrauth-auth-traps"
// DISABLE_CALLS-SAME: "ptrauth-indirect-gotos"
// DISABLE_CALLS-SAME: "ptrauth-returns"

// DISABLE_INDIRCT_GOTOS-NOT: ptrauth-indirect-gotos
// DISABLE_INDIRCT_GOTOS: attributes [[ATTR_MAIN]] = { {{.*}}"aarch64-jump-table-hardening"
// DISABLE_INDIRCT_GOTOS-SAME: "ptrauth-auth-traps"
// DISABLE_INDIRCT_GOTOS-SAME: "ptrauth-calls"
// DISABLE_INDIRCT_GOTOS-SAME: "ptrauth-returns"

// DISABLE_RETURNS-NOT: ptrauth-returns
// DISABLE_RETURNS: attributes [[ATTR_MAIN]] = { {{.*}}"aarch64-jump-table-hardening"
// DISABLE_RETURNS-SAME: "ptrauth-auth-traps"
// DISABLE_RETURNS-SAME: "ptrauth-calls"
// DISABLE_RETURNS-SAME: "ptrauth-indirect-gotos"

// NONE-NOT: ptrauth-returns
// NONE-NOT: aarch64-jump-table-hardening
// NONE-NOT: ptrauth-auth-traps
// NONE-NOT: ptrauth-calls
// NONE-NOT: ptrauth-indirect-gotos
