//@ ignore-lldb

// Test that line numbers are not replaced with those of the outermost expansion site when
// `#[collapse_debuginfo]` attribute us not used.

//@ compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc1[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc2[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc3[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc4[...]
// gdb-command:continue

fn one() {
    println!("one");
}
fn two() {
    println!("two");
}
fn three() {
    println!("three");
}
fn four() {
    println!("four");
}

macro_rules! outer {
    ($b:block) => {
        one(); // #loc1
        inner!();
        $b
    };
}

macro_rules! inner {
    () => {
        two(); // #loc2
    };
}

fn main() {
    let ret = 0; // #break
    outer!({
        three(); // #loc3
        four(); // #loc4
    });
    std::process::exit(ret);
}
