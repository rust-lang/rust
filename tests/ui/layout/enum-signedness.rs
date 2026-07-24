//@ only-64bit
#![feature(rustc_attrs)]
#![feature(never_type)]
#![crate_type = "lib"]

// When picking a representation for things that don't force a particular discriminant type,
// we look at how the value was actually written, which means that "equivalent" things
// actually need to come out different.

// The tests run only on 64-bit because we have to give things as `isize` in these cases.

#[rustc_dump_layout(largest_niche)]
#[repr(Rust)]
enum NegativeByteRust {
    //~^ ERROR: value: i8, valid_range: 155..=156
    A = -100,
    B = -101,
}

#[rustc_dump_layout(largest_niche)]
#[repr(C)]
enum NegativeByteC {
    //~^ ERROR: value: i32, valid_range: 4294967195..=4294967196
    A = -100,
    B = -101,
}

#[rustc_dump_layout(largest_niche)]
#[repr(Rust)]
enum PositiveByteRust {
    //~^ ERROR: value: u8, valid_range: 155..=156
    A = 256 - 100,
    B = 256 - 101,
}

#[rustc_dump_layout(largest_niche)]
#[repr(C)]
enum PositiveByteC {
    //~^ ERROR: value: u32, valid_range: 155..=156
    A = 256 - 100,
    B = 256 - 101,
}

#[rustc_dump_layout(largest_niche)]
#[repr(Rust)]
enum Negative32BitRust {
    //~^ ERROR: value: i32, valid_range: 0..=2147483648
    A = 0,
    B = i32::MIN as isize,
}

#[rustc_dump_layout(largest_niche)]
#[repr(C)]
enum Negative32BitC {
    //~^ ERROR: value: i32, valid_range: 0..=2147483648
    A = 0,
    B = i32::MIN as isize,
}

#[rustc_dump_layout(largest_niche)]
#[repr(Rust)]
enum Positive32BitRust {
    //~^ ERROR: value: u32, valid_range: 0..=2147483648
    A = 0,
    B = i32::MIN.cast_unsigned() as isize,
}

#[rustc_dump_layout(largest_niche)]
#[repr(C)]
enum Positive32BitC {
    //~^ ERROR: value: u32, valid_range: 0..=2147483648
    A = 0,
    B = i32::MIN.cast_unsigned() as isize,
}

#[rustc_dump_layout(largest_niche)]
#[repr(Rust)]
enum Negative64BitRust {
    //~^ ERROR: value: i64, valid_range: 9223372036854775808..=9223372036854775809
    A = i64::MIN as isize,
    B = i64::MIN as isize + 1,
}

#[rustc_dump_layout(largest_niche)]
#[repr(C)]
enum Negative64BitC {
    //~^ ERROR: value: i64, valid_range: 9223372036854775808..=9223372036854775809
    A = i64::MIN as isize,
    //~^ WARN: enum discriminant does not fit into C
    //~| WARN: previously accepted
    B = i64::MIN as isize + 1,
    //~^ WARN: enum discriminant does not fit into C
    //~| WARN: previously accepted
}

#[rustc_dump_layout(largest_niche)]
#[repr(Rust)]
enum Positive64BitRust {
    //~^ ERROR: value: u64, valid_range: 9223372036854775806..=9223372036854775807
    A = i64::MAX as isize - 1,
    B = i64::MAX as isize,
}

#[rustc_dump_layout(largest_niche)]
#[repr(C)]
enum Positive64BitC {
    //~^ ERROR: value: u64, valid_range: 9223372036854775806..=9223372036854775807
    A = i64::MAX as isize - 1,
    //~^ WARN: enum discriminant does not fit into C
    //~| WARN: previously accepted
    B = i64::MAX as isize,
    //~^ WARN: enum discriminant does not fit into C
    //~| WARN: previously accepted
}
