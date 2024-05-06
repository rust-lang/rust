//@ ignore-android: FIXME(#10381)
//@ ignore-windows
//@ compile-flags:-g

// gdb-command: run

// gdb-command: print slice
// gdbg-check: $1 = struct &[i32](size=3) = {0, 1, 2}
// gdbr-check: $1 = &[i32](size=3) = {0, 1, 2}

// gdb-command: print mut_slice
// gdbg-check: $2 = struct &mut [i32](size=4) = {2, 3, 5, 7}
// gdbr-check: $2 = &mut [i32](size=4) = {2, 3, 5, 7}

// gdb-command: print str_slice
// gdb-check: $3 = "string slice"

// gdb-command: print mut_str_slice
// gdb-check: $4 = "mutable string slice"

// lldb-command:run

// lldb-command:v slice
// lldb-check:(&[i32]) slice = size=3 { [0] = 0 [1] = 1 [2] = 2 }

// lldb-command:v mut_slice
// lldb-check:(&mut [i32]) mut_slice = size=4 { [0] = 2 [1] = 3 [2] = 5 [3] = 7 }

// lldb-command:v str_slice
// lldb-check:(&str) str_slice = "string slice" { data_ptr = [...] length = 12 }

// lldb-command:v mut_str_slice
// lldb-check:(&mut str) mut_str_slice = "mutable string slice" { data_ptr = [...] length = 20 }

fn b() {}

fn main() {
    let slice: &[i32] = &[0, 1, 2];
    let mut_slice: &mut [i32] = &mut [2, 3, 5, 7];

    let str_slice: &str = "string slice";
    let mut mut_str_slice_buffer = String::from("mutable string slice");
    let mut_str_slice: &mut str = mut_str_slice_buffer.as_mut_str();

    b(); // #break
}
