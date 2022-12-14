// ignore-lldb
#![feature(collapse_debuginfo)]

// Test that line numbers are replaced with those of the outermost expansion site when the
// `collapse_debuginfo` feature is active and the attribute is provided.

// compile-flags:-g

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

#[collapse_debuginfo]
macro_rules! outer {
    ($b:block) => {
        one();
        inner!();
        $b
    };
}

#[collapse_debuginfo]
macro_rules! inner {
    () => {
        two();
    };
}

fn main() {
    let ret = 0; // #break
    outer!({ // #loc1
        three(); // #loc2
        four(); // #loc3
    });
    std::process::exit(ret);
}
