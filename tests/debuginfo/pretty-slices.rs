//@ ignore-android: FIXME(#10381)
//@ ignore-windows-gnu: #128981
//@ compile-flags:-g
//@ ignore-backends: gcc

//@ gdb-command: run

//@ gdb-command: print slice
//@ gdb-check: $1 = &[i32](size=3) = {0, 1, 2}

//@ gdb-command: print mut_slice
//@ gdb-check: $2 = &mut [i32](size=4) = {2, 3, 5, 7}

//@ gdb-command: print str_slice
//@ gdb-check: $3 = "string slice"

//@ gdb-command: print mut_str_slice
//@ gdb-check: $4 = "mutable string slice"

//@ lldb-command:run

//@ lldb-repr:slice
//@ lldb-repr:mut_slice
//@ lldb-repr:str_slice
//@ lldb-repr:mut_str_slice

fn b() {}

fn main() {
    let slice: &[i32] = &[0, 1, 2];
    let mut_slice: &mut [i32] = &mut [2, 3, 5, 7];

    let str_slice: &str = "string slice";
    let mut mut_str_slice_buffer = String::from("mutable string slice");
    let mut_str_slice: &mut str = mut_str_slice_buffer.as_mut_str();

    b(); // #break
}
