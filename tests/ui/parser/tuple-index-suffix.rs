//! See #60210.
//!
//! Check that we hard error on invalid suffixes in tuple indexing subexpressions and struct numeral
//! field names, modulo carve-outs for `{i,u}{32,usize}` at warning level to mitigate ecosystem
//! impact.

struct X(i32,i32,i32);

fn main() {
    let tup_struct = X(1, 2, 3);
    let invalid_tup_struct_suffix = tup_struct.0suffix;
    //~^ ERROR suffixes on a tuple index are invalid
    let carve_out_tup_struct_suffix = tup_struct.0i32;
    //~^ WARN suffixes on a tuple index are invalid

    let tup = (1, 2, 3);
    let invalid_tup_suffix = tup.0suffix;
    //~^ ERROR suffixes on a tuple index are invalid
    let carve_out_tup_suffix = tup.0u32;
    //~^ WARN suffixes on a tuple index are invalid

    numeral_struct_field_name_suffix_invalid();
    numeral_struct_field_name_suffix_carve_out();
}

// Very limited carve outs as a ecosystem impact mitigation implemented in #60186. *Only*
// `{i,u}{32,usize}` suffixes are temporarily accepted.
fn carve_outs() {
    // Ok, only pseudo-FCW warnings.

    let carve_out_i32 = (42,).0i32;     //~ WARN suffixes on a tuple index are invalid
    let carve_out_i32 = (42,).0u32;     //~ WARN suffixes on a tuple index are invalid
    let carve_out_isize = (42,).0isize; //~ WARN suffixes on a tuple index are invalid
    let carve_out_usize = (42,).0usize; //~ WARN suffixes on a tuple index are invalid

    // Not part of the carve outs!
    let error_i8 = (42,).0i8;      //~ ERROR suffixes on a tuple index are invalid
    let error_u8 = (42,).0u8;      //~ ERROR suffixes on a tuple index are invalid
    let error_i16 = (42,).0i16;    //~ ERROR suffixes on a tuple index are invalid
    let error_u16 = (42,).0u16;    //~ ERROR suffixes on a tuple index are invalid
    let error_i64 = (42,).0i64;    //~ ERROR suffixes on a tuple index are invalid
    let error_u64 = (42,).0u64;    //~ ERROR suffixes on a tuple index are invalid
    let error_i128 = (42,).0i128;  //~ ERROR suffixes on a tuple index are invalid
    let error_u128 = (42,).0u128;  //~ ERROR suffixes on a tuple index are invalid
}

fn numeral_struct_field_name_suffix_invalid() {
    let invalid_struct_name = X { 0suffix: 0, 1: 1, 2: 2 };
    //~^ ERROR suffixes on a tuple index are invalid
    match invalid_struct_name {
        X { 0suffix: _, .. } => {}
        //~^ ERROR suffixes on a tuple index are invalid
    }
}

fn numeral_struct_field_name_suffix_carve_out() {
    let carve_out_struct_name = X { 0u32: 0, 1: 1, 2: 2 };
    //~^ WARN suffixes on a tuple index are invalid
    match carve_out_struct_name {
        X { 0u32: _, .. } => {}
        //~^ WARN suffixes on a tuple index are invalid
    }
}

// Unfortunately, it turns out `std::mem::offset_of!` uses the same expect suffix code path.
fn offset_of_suffix() {
    #[repr(C)]
    pub struct Struct<T>(u8, T);

    // Carve outs
    assert_eq!(std::mem::offset_of!(Struct<u32>, 0usize), 0);
    //~^ WARN suffixes on a tuple index are invalid

    // Not part of carve outs
    assert_eq!(std::mem::offset_of!(Struct<u32>, 0u8), 0);
    //~^ ERROR suffixes on a tuple index are invalid
}
