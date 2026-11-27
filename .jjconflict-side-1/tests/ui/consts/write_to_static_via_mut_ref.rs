static OH_NO: &mut i32 = &mut 42; //~ ERROR mutable borrows of temporaries
fn main() {
    assert_eq!(*OH_NO, 42);
    *OH_NO = 43; //~ ERROR cannot assign to `*OH_NO`, as `OH_NO` is an immutable static
}
