#![feature(transparent_unions)]
#![warn(clippy::default_union_representation)]
#![allow(clippy::repr_packed_without_abi)]

union NoAttribute {
    //~^ default_union_representation
    a: i32,
    b: u32,
}

#[repr(C)]
union ReprC {
    a: i32,
    b: u32,
}

#[repr(packed)]
union ReprPacked {
    //~^ default_union_representation
    a: i32,
    b: u32,
}

#[repr(C, packed)]
union ReprCPacked {
    a: i32,
    b: u32,
}

#[repr(C, align(32))]
union ReprCAlign {
    a: i32,
    b: u32,
}

#[repr(align(32))]
union ReprAlign {
    //~^ default_union_representation
    a: i32,
    b: u32,
}

union SingleZST {
    f0: (),
}
union ZSTsAndField1 {
    f0: u32,
    f1: (),
    f2: (),
    f3: (),
}
union ZSTsAndField2 {
    f0: (),
    f1: (),
    f2: u32,
    f3: (),
}
union ZSTAndTwoFields {
    //~^ default_union_representation
    f0: u32,
    f1: u64,
    f2: (),
}

#[repr(C)]
union CZSTAndTwoFields {
    f0: u32,
    f1: u64,
    f2: (),
}

#[repr(transparent)]
union ReprTransparent {
    a: i32,
}

#[repr(transparent)]
union ReprTransparentZST {
    a: i32,
    b: (),
}

fn main() {}
