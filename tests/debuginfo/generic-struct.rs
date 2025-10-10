//@ compile-flags:-g
//@ disable-gdb-pretty-printers

// === GDB TESTS ===================================================================================

// gdb-command:run

// gdb-command:print int_int
// gdb-check:$1 = generic_struct::AGenericStruct<i32, i32> {key: 0, value: 1}
// gdb-command:print int_float
// gdb-check:$2 = generic_struct::AGenericStruct<i32, f64> {key: 2, value: 3.5}
// gdb-command:print float_int
// gdb-check:$3 = generic_struct::AGenericStruct<f64, i32> {key: 4.5, value: 5}
// gdb-command:print float_int_float
// gdb-check:$4 = generic_struct::AGenericStruct<f64, generic_struct::AGenericStruct<i32, f64>> {key: 6.5, value: generic_struct::AGenericStruct<i32, f64> {key: 7, value: 8.5}}

// === LLDB TESTS ==================================================================================

// lldb-command:run

// lldb-command:v int_int
// lldb-check:[...]AGenericStruct<int, int>) int_int = { key = 0 value = 1 }
// lldb-command:v int_float
// lldb-check:[...]AGenericStruct<int, double>) int_float = { key = 2 value = 3.5 }
// lldb-command:v float_int
// lldb-check:[...]AGenericStruct<double, int>) float_int = { key = 4.5 value = 5 }

// lldb-command:v float_int_float
// lldb-check:[...]AGenericStruct<double, generic_struct::AGenericStruct<int, double> >) float_int_float = { key = 6.5 value = { key = 7 value = 8.5 } }

// === CDB TESTS ===================================================================================

// cdb-command:g

// cdb-command:dx int_int
// cdb-check:int_int          [Type: generic_struct::AGenericStruct<i32,i32>]
// cdb-check:[...]key              : 0 [Type: int]
// cdb-check:[...]value            : 1 [Type: int]
// cdb-command:dx int_float
// cdb-check:int_float        [Type: generic_struct::AGenericStruct<i32,f64>]
// cdb-check:[...]key              : 2 [Type: int]
// cdb-check:[...]value            : 3.500000 [Type: double]
// cdb-command:dx float_int
// cdb-check:float_int        [Type: generic_struct::AGenericStruct<f64,i32>]
// cdb-check:[...]key              : 4.500000 [Type: double]
// cdb-check:[...]value            : 5 [Type: int]
// cdb-command:dx float_int_float
// cdb-check:float_int_float  [Type: generic_struct::AGenericStruct<f64,generic_struct::AGenericStruct<i32,f64> >]
// cdb-check:[...]key              : 6.500000 [Type: double]
// cdb-check:[...]value            [Type: generic_struct::AGenericStruct<i32,f64>]


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
