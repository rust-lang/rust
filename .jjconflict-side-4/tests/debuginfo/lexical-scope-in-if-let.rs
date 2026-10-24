//@ revisions: msvc

// On MSVC, this test requires LLDB to be able to read `S_DEFRANGE_REGISTER_REL_INDIR` PDB nodes,
// which is only possible on lldb 23.1+
//@ [msvc] only-msvc
//@ [msvc] min-llvm-lldb-version: 23.1.0

//@ compile-flags:-g
//@ ignore-backends: gcc

// === GDB TESTS ==================================================================================

//@ gdb-command:run
//@ gdb-command:info locals
//@ gdb-check:a = 123

//@ gdb-command:continue
//@ gdb-command:info locals
//@ gdb-check:x = 42
//@ gdb-check:a = 123

//@ gdb-command:continue
//@ gdb-command:info locals
//@ gdb-check:y = true
//@ gdb-check:b = 456
//@ gdb-check:x = 42
//@ gdb-check:a = 123

//@ gdb-command:continue
//@ gdb-command:info locals
//@ gdb-check:z = 10
//@ gdb-check:c = 789
//@ gdb-check:y = true
//@ gdb-check:b = 456
//@ gdb-check:x = 42
//@ gdb-check:a = 123

// === LLDB TESTS =================================================================================

// Note: when printing full frames on msvc, LLDB does not ommit variables that are out of scope, so
// we add a wildcard at the end to make sure the e.g. "(int) x = <variable not available>" is
// accounted for

//@ lldb-command:run
//@ lldb-command:frame variable
//@ lldb-check:(int) a = 123[...]

//@ lldb-command:continue
//@ lldb-command:frame variable
//@ lldb-check:(int) a = 123 (int) x = 42[...]

//@ lldb-command:continue
//@ lldb-command:frame variable
//@ lldb-check:(int) a = 123 (int) x = 42 (int) b = 456 (bool) y = true[...]

//@ lldb-command:continue
//@ lldb-command:frame variable
//@ lldb-check:(int) a = 123 (int) x = 42 (int) b = 456 (bool) y = true (int) c = 789 (int) z = 10

// === CDB TESTS ==================================================================================

// Note: `/n` causes the the output to be sorted to avoid depending on the order in PDB which may
// be arbitrary.

//@ cdb-command: g
//@ cdb-command: dv /n
//@ cdb-check:[...]a = 0n123

//@ cdb-command: g
//@ cdb-command: dv /n
//@ cdb-check:[...]a = 0n123
//@ cdb-check:[...]x = 0n42

//@ cdb-command: g
//@ cdb-command: dv /n
//@ cdb-check:[...]a = 0n123
//@ cdb-check:[...]b = 0n456
//@ cdb-check:[...]x = 0n42
//@ cdb-check:[...]y = true

//@ cdb-command: g
//@ cdb-command: dv /n
//@ cdb-check:[...]a = 0n123
//@ cdb-check:[...]b = 0n456
//@ cdb-check:[...]c = 0n789
//@ cdb-check:[...]x = 0n42
//@ cdb-check:[...]y = true
//@ cdb-check:[...]z = 0n10

fn main() {
    let a = id(123);

    zzz(); // #break

    if let Some(x) = id(Some(42)) {
        zzz(); // #break

        let b = id(456);

        if let Ok(y) = id::<Result<bool, ()>>(Ok(true)) {
            zzz(); // #break

            let c = id(789);

            if let (z, 42) = id((10, 42)) {
                zzz(); // #break
            }
        }
    }
}

#[inline(never)]
fn id<T>(value: T) -> T {
    value
}

fn zzz() {}
