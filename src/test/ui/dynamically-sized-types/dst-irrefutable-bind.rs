// run-pass
#![feature(unsized_tuple_coercion)]

struct Test<T: ?Sized>(T);

fn main() {
    let x = Test([1,2,3]);
    let x : &Test<[i32]> = &x;

    let & ref _y = x;

    // Make sure binding to a fat pointer behind a reference
    // still works
    let slice = &[1,2,3];
    let x = Test(&slice);
    let Test(&_slice) = x;


    let x = (10, [1,2,3]);
    let x : &(i32, [i32]) = &x;

    let & ref _y = x;

    let slice = &[1,2,3];
    let x = (10, &slice);
    let (_, &_slice) = x;
}
