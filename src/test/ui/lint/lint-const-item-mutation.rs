// check-pass

struct MyStruct {
    field: bool,
    inner_array: [char; 1],
    raw_ptr: *mut u8
}
impl MyStruct {
    fn use_mut(&mut self) {}
}

const ARRAY: [u8; 1] = [25];
const MY_STRUCT: MyStruct = MyStruct { field: true, inner_array: ['a'], raw_ptr: 2 as *mut u8 };
const RAW_PTR: *mut u8 = 1 as *mut u8;

fn main() {
    ARRAY[0] = 5; //~ WARN attempting to modify
    MY_STRUCT.field = false; //~ WARN attempting to modify
    MY_STRUCT.inner_array[0] = 'b'; //~ WARN attempting to modify
    MY_STRUCT.use_mut(); //~ WARN taking
    &mut MY_STRUCT; //~ WARN taking
    (&mut MY_STRUCT).use_mut(); //~ WARN taking

    // Test that we don't warn when writing through
    // a raw pointer
    // This is U.B., but this test is check-pass,
    // so this never actually executes
    unsafe {
        *RAW_PTR = 0;
        *MY_STRUCT.raw_ptr = 0;
    }
}
