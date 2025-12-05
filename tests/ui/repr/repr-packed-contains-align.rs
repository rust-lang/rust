#![allow(dead_code)]

#[repr(align(16))]
#[derive(Clone, Copy)]
struct SA(i32);

#[derive(Clone, Copy)]
struct SB(SA);

#[repr(align(16))]
#[derive(Clone, Copy)]
union UA {
    i: i32
}

#[derive(Clone, Copy)]
union UB {
    a: UA
}

#[repr(packed)]
struct SC(SA); //~ ERROR: packed type cannot transitively contain a `#[repr(align)]` type

#[repr(packed)]
struct SD(SB); //~ ERROR: packed type cannot transitively contain a `#[repr(align)]` type

#[repr(packed)]
struct SE(UA); //~ ERROR: packed type cannot transitively contain a `#[repr(align)]` type

#[repr(packed)]
struct SF(UB); //~ ERROR: packed type cannot transitively contain a `#[repr(align)]` type

#[repr(packed)]
union UC { //~ ERROR: packed type cannot transitively contain a `#[repr(align)]` type
    a: UA
}

#[repr(packed)]
union UD { //~ ERROR: packed type cannot transitively contain a `#[repr(align)]` type
    n: UB
}

#[repr(packed)]
union UE { //~ ERROR: packed type cannot transitively contain a `#[repr(align)]` type
    a: SA
}

#[repr(packed)]
union UF { //~ ERROR: packed type cannot transitively contain a `#[repr(align)]` type
    n: SB
}

fn main() {}
