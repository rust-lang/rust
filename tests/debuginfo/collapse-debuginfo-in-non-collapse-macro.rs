//@ ignore-lldb

// Test that statement, skipped/added/reordered by macros, is correctly processed in debuginfo.
// When nested macros instantiations are tagged with collapse_debuginfo attribute,
// debug info should be corrected to the first outer macro instantiation
// without collapse_debuginfo attribute.

//@ compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc_rem_call1[...]
// gdb-command:step
// gdb-command:frame
// gdb-check:[...]#loc_call1_pre[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc_in_proxy[...]
// gdb-command:next 2
// gdb-check:[...]#loc_rem_call3[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc_add_call1[...]
// gdb-command:step
// gdb-command:frame
// gdb-check:[...]#loc_call1_pre[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc_in_proxy[...]
// gdb-command:next 2
// gdb-check:[...]#loc_add_macro[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc_add_call3[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc_reorder_call2[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc_reorder_call3[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc_reorder_call1[...]
// gdb-command:step
// gdb-command:frame
// gdb-check:[...]#loc_call1_pre[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc_in_proxy[...]
// gdb-command:next 2
// gdb-command:frame
// gdb-command:continue

#[inline(never)]
fn myprintln_impl(text: &str) {
    println!("{}", text)
}

#[collapse_debuginfo(yes)]
macro_rules! myprintln {
    ($($arg:tt)*) => {{
        myprintln_impl($($arg)*);
    }};
}

macro_rules! proxy_println {
    ($($arg:tt)*) => {{
        myprintln!($($arg)*); // #loc_in_proxy
    }};
}

// Macro accepts 3 statements and removes the 2nd statement
macro_rules! remove_second_statement {
    ($s1:stmt; $s2:stmt; $s3:stmt;) => { $s1 $s3 }
}

macro_rules! add_second_statement {
    ($s1:stmt; $s3:stmt;) => {
        $s1
        call2(); // #loc_add_macro
        $s3
    }
}

macro_rules! reorder_statements {
    ($s1:stmt; $s2:stmt; $s3:stmt;) => { $s2 $s3 $s1 }
}

fn call1() {
    let rv = 0; // #loc_call1_pre
    proxy_println!("one"); // #loc_call1
}

fn call2() {
    proxy_println!("two"); // #loc_call2
}

fn call3() {
    proxy_println!("three"); // #loc_call3
}

fn main() {
    let ret = 0; // #break, step should go to call1
    remove_second_statement! { // #loc_rem_hdr
        call1(); // #loc_rem_call1
        call2(); // #loc_rem_call2
        call3(); // #loc_rem_call3
    }
    add_second_statement! { // #loc_add_hdr
        call1(); // #loc_add_call1
        call3(); // #loc_add_call3
    }
    reorder_statements! { // #loc_reorder_hdr
        call1(); // #loc_reorder_call1
        call2(); // #loc_reorder_call2
        call3(); // #loc_reorder_call3
    }
    std::process::exit(ret); // #loc_exit
}
