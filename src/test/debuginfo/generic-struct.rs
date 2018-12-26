// ignore-tidy-linelength

// Some versions of the non-rust-enabled LLDB print the wrong generic
// parameter type names in this test.
// rust-lldb

// compile-flags:-g

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print int_int
// gdbg-check:$1 = {key = 0, value = 1}
// gdbr-check:$1 = generic_struct::AGenericStruct<i32, i32> {key: 0, value: 1}
// gdb-command:print int_float
// gdbg-check:$2 = {key = 2, value = 3.5}
// gdbr-check:$2 = generic_struct::AGenericStruct<i32, f64> {key: 2, value: 3.5}
// gdb-command:print float_int
// gdbg-check:$3 = {key = 4.5, value = 5}
// gdbr-check:$3 = generic_struct::AGenericStruct<f64, i32> {key: 4.5, value: 5}
// gdb-command:print float_int_float
// gdbg-check:$4 = {key = 6.5, value = {key = 7, value = 8.5}}
// gdbr-check:$4 = generic_struct::AGenericStruct<f64, generic_struct::AGenericStruct<i32, f64>> {key: 6.5, value: generic_struct::AGenericStruct<i32, f64> {key: 7, value: 8.5}}

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:print int_int
// lldbg-check:[...]$0 = AGenericStruct<i32, i32> { key: 0, value: 1 }
// lldbr-check:(generic_struct::AGenericStruct<i32, i32>) int_int = AGenericStruct<i32, i32> { key: 0, value: 1 }
// lldb-command:print int_float
// lldbg-check:[...]$1 = AGenericStruct<i32, f64> { key: 2, value: 3.5 }
// lldbr-check:(generic_struct::AGenericStruct<i32, f64>) int_float = AGenericStruct<i32, f64> { key: 2, value: 3.5 }
// lldb-command:print float_int
// lldbg-check:[...]$2 = AGenericStruct<f64, i32> { key: 4.5, value: 5 }
// lldbr-check:(generic_struct::AGenericStruct<f64, i32>) float_int = AGenericStruct<f64, i32> { key: 4.5, value: 5 }

// lldb-command:print float_int_float
// lldbg-check:[...]$3 = AGenericStruct<f64, generic_struct::AGenericStruct<i32, f64>> { key: 6.5, value: AGenericStruct<i32, f64> { key: 7, value: 8.5 } }
// lldbr-check:(generic_struct::AGenericStruct<f64, generic_struct::AGenericStruct<i32, f64>>) float_int_float = AGenericStruct<f64, generic_struct::AGenericStruct<i32, f64>> { key: 6.5, value: AGenericStruct<i32, f64> { key: 7, value: 8.5 } }


#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

struct AGenericStruct<TKey, TValue> {
    key: TKey,
    value: TValue
}

fn main() {

    let int_int = AGenericStruct { key: 0, value: 1 };
    let int_float = AGenericStruct { key: 2, value: 3.5f64 };
    let float_int = AGenericStruct { key: 4.5f64, value: 5 };
    let float_int_float = AGenericStruct {
        key: 6.5f64,
        value: AGenericStruct { key: 7, value: 8.5f64 },
    };

    zzz(); // #break
}

fn zzz() { () }
