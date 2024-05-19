//@ ignore-lldb

// Test that statement, skipped/added/reordered by macros, is correctly processed in debuginfo.
// Performed step-over and step-into debug stepping through call statements.

//@ compile-flags:-g -C collapse-macro-debuginfo=yes

// === GDB TESTS ===================================================================================

// gdb-command:run
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc_rem1_call1[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc_rem1_call3[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc_rem2_call1[...]
// gdb-command:step
// gdb-command:frame
// gdb-check:[...]#loc_call1[...]
// gdb-command:next 2
// gdb-check:[...]#loc_rem2_call3[...]
// gdb-command:step 2
// gdb-command:frame
// gdb-check:[...]#loc_call3_println[...]
// gdb-command:next 3
// gdb-command:frame
// gdb-check:[...]#loc_after_rem[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc_add1_call1[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc_add1_hdr[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc_add1_call3[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc_add2_call1[...]
// gdb-command:step
// gdb-command:frame
// gdb-check:[...]#loc_call1[...]
// gdb-command:next 2
// gdb-check:[...]#loc_add2_hdr[...]
// gdb-command:step
// gdb-command:frame
// gdb-check:[...]#loc_call2[...]
// gdb-command:next 2
// gdb-command:frame
// gdb-check:[...]#loc_add2_call3[...]
// gdb-command:step 2
// gdb-command:frame
// gdb-check:[...]#loc_call3_println[...]
// gdb-command:next 3
// gdb-command:frame
// gdb-check:[...]#loc_reorder1_call2[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc_reorder1_call3[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc_reorder1_call1[...]
// gdb-command:next
// gdb-command:frame
// gdb-check:[...]#loc_reorder2_call2[...]
// gdb-command:step
// gdb-command:frame
// gdb-check:[...]#loc_call2[...]
// gdb-command:next 2
// gdb-command:frame
// gdb-check:[...]#loc_reorder2_call3[...]
// gdb-command:step
// gdb-command:frame
// gdb-check:[...]#loc_call3[...]
// gdb-command:step
// gdb-command:frame
// gdb-check:[...]#loc_call3_println[...]
// gdb-command:next 3
// gdb-command:frame
// gdb-check:[...]#loc_reorder2_call1[...]
// gdb-command:step
// gdb-command:frame
// gdb-check:[...]#loc_call1[...]
// gdb-command:next 2
// gdb-command:continue

#[inline(never)]
fn myprintln_impl(text: &str) {
    println!("{}", text)
}

macro_rules! myprintln {
    ($($arg:tt)*) => {{
        myprintln_impl($($arg)*);
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
    myprintln!("one"); // #loc_call1
}

fn call2() {
    myprintln!("two"); // #loc_call2
}

fn call3() {
    (||{
        myprintln!("three") // #loc_call3_println
    })(); // #loc_call3
}

fn main() {
    let ret = 0; // #break, step should go to call1
    remove_second_statement! { // #loc_rem1_hdr
        call1(); // #loc_rem1_call1, breakpoint should set to call1, step should go call3
        call2(); // #loc_rem1_call2, breakpoint should set to call3
        call3(); // #loc_rem1_call3
    }
    remove_second_statement! { // #loc_rem2_hdr
        call1(); // #loc_rem2_call1, breakpoint should set to call1, step should go call3
        call2(); // #loc_rem2_call2, breakpoint should set to call3
        call3(); // #loc_rem2_call3, breakpoint should set to call3
    }
    myprintln!("After remove_second_statement test"); // #loc_after_rem

    add_second_statement! { // #loc_add1_hdr
        call1(); // #loc_add1_call1
        call3(); // #loc_add1_call3
    }
    add_second_statement! { // #loc_add2_hdr
        call1(); // #loc_add2_call1
        call3(); // #loc_add2_call3
    }

    reorder_statements! { // #loc_reorder1_hdr
        call1(); // #loc_reorder1_call1
        call2(); // #loc_reorder1_call2
        call3(); // #loc_reorder1_call3
    }
    reorder_statements! { // #loc_reorder2_hdr
        call1(); // #loc_reorder2_call1
        call2(); // #loc_reorder2_call2
        call3(); // #loc_reorder2_call3
    }

    std::process::exit(ret); // #loc_exit
}
