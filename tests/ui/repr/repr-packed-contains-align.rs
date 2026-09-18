#![allow(dead_code)]

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct SA(i32);

#[repr(align(16))]
#[derive(Clone, Copy)]
struct SARust(i32);

#[repr(C)]
#[derive(Clone, Copy)]
struct SB(SA);

#[repr(C, align(16))]
#[derive(Clone, Copy)]
union UA {
    i: i32
}

#[repr(C)]
#[derive(Clone, Copy)]
union UB {
    a: UA
}

#[repr(C, packed)]
struct SC(SA); //~ ERROR: packed type cannot transitively contain a `#[repr(align)]` type

#[repr(C, packed)]
struct SD(SB); //~ ERROR: packed type cannot transitively contain a `#[repr(align)]` type

#[repr(C, packed)]
struct SE(UA); //~ ERROR: packed type cannot transitively contain a `#[repr(align)]` type

#[repr(C, packed)]
struct SF(UB); //~ ERROR: packed type cannot transitively contain a `#[repr(align)]` type

#[repr(C, packed)]
union UC { //~ ERROR: packed type cannot transitively contain a `#[repr(align)]` type
    a: UA
}

#[repr(C, packed)]
union UD { //~ ERROR: packed type cannot transitively contain a `#[repr(align)]` type
    n: UB
}

#[repr(C, packed)]
union UE { //~ ERROR: packed type cannot transitively contain a `#[repr(align)]` type
    a: SA
}

#[repr(C, packed)]
union UF { //~ ERROR: packed type cannot transitively contain a `#[repr(align)]` type
    n: SB
}

#[repr(packed)]
struct SG(SA); // outer type not `repr(C)`, no lint
#[repr(C, packed)]
struct SH(SARust); // inner type not `repr(C)`, no lint



fn main() {}
