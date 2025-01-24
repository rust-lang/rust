//@ ignore-android: FIXME(#10381)
//@ ignore-windows-gnu: #128981
//@ compile-flags:-g

// gdb-command: run

// gdb-command: print slice
// gdb-check: $1 = &[i32](size=3) = {0, 1, 2}

// gdb-command: print mut_slice
// gdb-check: $2 = &mut [i32](size=4) = {2, 3, 5, 7}

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
// lldb-check:(&str) str_slice = "string slice" { [0] = 's' [1] = 't' [2] = 'r' [3] = 'i' [4] = 'n' [5] = 'g' [6] = ' ' [7] = 's' [8] = 'l' [9] = 'i' [10] = 'c' [11] = 'e' }

// lldb-command:v mut_str_slice
// lldb-check:(&mut str) mut_str_slice = "mutable string slice" { [0] = 'm' [1] = 'u' [2] = 't' [3] = 'a' [4] = 'b' [5] = 'l' [6] = 'e' [7] = ' ' [8] = 's' [9] = 't' [10] = 'r' [11] = 'i' [12] = 'n' [13] = 'g' [14] = ' ' [15] = 's' [16] = 'l' [17] = 'i' [18] = 'c' [19] = 'e' }

fn b() {}

fn main() {
    let slice: &[i32] = &[0, 1, 2];
    let mut_slice: &mut [i32] = &mut [2, 3, 5, 7];

    let str_slice: &str = "string slice";
    let mut mut_str_slice_buffer = String::from("mutable string slice");
    let mut_str_slice: &mut str = mut_str_slice_buffer.as_mut_str();

    b(); // #break
}
