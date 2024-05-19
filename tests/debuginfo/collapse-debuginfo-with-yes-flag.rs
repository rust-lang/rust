//@ ignore-lldb

// Test that line numbers are replaced with those of the outermost expansion site when the
// the command line flag is passed.

//@ compile-flags:-g -C collapse_macro_debuginfo=yes

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

macro_rules! outer {
    ($b:block) => {
        one();
        inner!();
        $b
    };
}

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
