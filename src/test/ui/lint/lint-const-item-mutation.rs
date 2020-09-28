// check-pass

struct MyStruct {
    field: bool,
    inner_array: [char; 1],
}
impl MyStruct {
    fn use_mut(&mut self) {}
}

const ARRAY: [u8; 1] = [25];
const MY_STRUCT: MyStruct = MyStruct { field: true, inner_array: ['a'] };

fn main() {
    ARRAY[0] = 5; //~ WARN attempting to modify
    MY_STRUCT.field = false; //~ WARN attempting to modify
    MY_STRUCT.inner_array[0] = 'b'; //~ WARN attempting to modify
    MY_STRUCT.use_mut(); //~ WARN taking
    &mut MY_STRUCT; //~ WARN taking
    (&mut MY_STRUCT).use_mut(); //~ WARN taking
}
