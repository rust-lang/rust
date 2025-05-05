// Checks that values for `-Clink-self-contained` other than the blanket enable/disable and
// `+/-linker` require `-Zunstable-options`.

//@ revisions: crto libc unwind sanitizers mingw
//@ [crto] compile-flags: -Clink-self-contained=+crto
//@ [libc] compile-flags: -Clink-self-contained=-libc
//@ [unwind] compile-flags: -Clink-self-contained=+unwind
//@ [sanitizers] compile-flags: -Clink-self-contained=-sanitizers
//@ [mingw] compile-flags: -Clink-self-contained=+mingw

fn main() {}

//[crto]~? ERROR only `-C link-self-contained` values `y`/`yes`/`on`/`n`/`no`/`off`/`-linker`/`+linker` are stable
//[libc]~? ERROR only `-C link-self-contained` values `y`/`yes`/`on`/`n`/`no`/`off`/`-linker`/`+linker` are stable
//[unwind]~? ERROR only `-C link-self-contained` values `y`/`yes`/`on`/`n`/`no`/`off`/`-linker`/`+linker` are stable
//[sanitizers]~? ERROR only `-C link-self-contained` values `y`/`yes`/`on`/`n`/`no`/`off`/`-linker`/`+linker` are stable
//[mingw]~? ERROR only `-C link-self-contained` values `y`/`yes`/`on`/`n`/`no`/`off`/`-linker`/`+linker` are stable
