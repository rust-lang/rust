//@ ignore-msvc: https://github.com/llvm/llvm-project/issues/221696
// Once the above is resolved, this test will still require the below revision

//@ revisions: msvc

// On MSVC, this test requires LLDB to be able to read `S_DEFRANGE_REGISTER_REL_INDIR` PDB nodes,
// which is only possible on lldb 23.1+
//@ [msvc] only-msvc
//@ [msvc] min-llvm-lldb-version: 23.1.0

//@ compile-flags:-g
//@ disable-gdb-pretty-printers
//@ ignore-backends: gcc

// === GDB TESTS ===================================================================================

//@ gdb-command:run

//@ gdb-command:print x
//@ gdb-check:$1 = false
//@ gdb-command:continue

//@ gdb-command:print x
//@ gdb-check:$2 = false
//@ gdb-command:continue

//@ gdb-command:print x
//@ gdb-check:$3 = 10
//@ gdb-command:continue

//@ gdb-command:print x
//@ gdb-check:$4 = 10
//@ gdb-command:continue

//@ gdb-command:print x
//@ gdb-check:$5 = 10.5
//@ gdb-command:continue

//@ gdb-command:print x
//@ gdb-check:$6 = 10
//@ gdb-command:continue

//@ gdb-command:print x
//@ gdb-check:$7 = false
//@ gdb-command:continue


// === LLDB TESTS ==================================================================================

//@ lldb-command:run

//@ lldb-command:v x
//@ lldb-check:[...] false
//@ lldb-command:continue

//@ lldb-command:v x
//@ lldb-check:[...] false
//@ lldb-command:continue

//@ lldb-command:v x
//@ lldb-check:[...] 10
//@ lldb-command:continue

//@ lldb-command:v x
//@ lldb-check:[...] 10
//@ lldb-command:continue

//@ lldb-command:v x
//@ lldb-check:[...] 10.5
//@ lldb-command:continue

//@ lldb-command:v x
//@ lldb-check:[...] 10
//@ lldb-command:continue

//@ lldb-command:v x
//@ lldb-check:[...] false
//@ lldb-command:continue


fn main() {
    let x = false;

    zzz(); // #break
    sentinel();

    {
        zzz(); // #break
        sentinel();

        let x = 10;

        zzz(); // #break
        sentinel();

        {
            zzz(); // #break
            sentinel();

            let x = 10.5f64;

            zzz(); // #break
            sentinel();
        }

        zzz(); // #break
        sentinel();
    }

    zzz(); // #break
    sentinel();
}

fn zzz() {()}
fn sentinel() {()}
