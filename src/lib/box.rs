/*
Module: box
*/


export ptr_eq;

/*
Function: ptr_eq

Determine if two shared boxes point to the same object
*/
fn ptr_eq<T>(a: @T, b: @T) -> bool {
    let a_ptr: uint = unsafe::reinterpret_cast(a);
    let b_ptr: uint = unsafe::reinterpret_cast(b);
    ret a_ptr == b_ptr;
}
