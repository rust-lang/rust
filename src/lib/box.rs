
export ptr_eq;

fn ptr_eq[T](&@T a, &@T b) -> bool {
    let uint a_ptr = unsafe::reinterpret_cast(a);
    let uint b_ptr = unsafe::reinterpret_cast(b);
    ret a_ptr == b_ptr;
}
