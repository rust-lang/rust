struct MyStruct { field: usize }
const STRUCT: MyStruct = MyStruct { field: 42 };

fn main() {
    let a: [isize; STRUCT.nonexistent_field];
    //~^ no field `nonexistent_field` on type `MyStruct`
}
