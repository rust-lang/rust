//! Regression test for both the original regression in #59418 where invalid suffixes in indexing
//! positions were accidentally accepted, and also for the removal of the temporary carve out that
//! mitigated ecosystem impact following trying to reject #59418 (this was implemented as a FCW
//! tracked in #60210).
//!
//! Check that we hard error on invalid suffixes in tuple indexing subexpressions and struct numeral
//! field names.

struct X(i32,i32,i32);

fn main() {
    let tup_struct = X(1, 2, 3);
    let invalid_tup_struct_suffix = tup_struct.0suffix;
    //~^ ERROR suffixes on a tuple index are invalid
    let previous_carve_out_tup_struct_suffix = tup_struct.0i32;
    //~^ ERROR suffixes on a tuple index are invalid

    let tup = (1, 2, 3);
    let invalid_tup_suffix = tup.0suffix;
    //~^ ERROR suffixes on a tuple index are invalid
    let previous_carve_out_tup_suffix = tup.0u32;
    //~^ ERROR suffixes on a tuple index are invalid

    numeral_struct_field_name_suffix_invalid();
    numeral_struct_field_name_suffix_previous_carve_out();
}

// Previously, there were very limited carve outs as a ecosystem impact mitigation implemented in
// #60186. *Only* `{i,u}{32,usize}` suffixes were temporarily accepted. Now, they all hard error.
fn previous_carve_outs() {
    // Previously temporarily accepted by a pseudo-FCW (#60210), now hard error.

    let previous_carve_out_i32 = (42,).0i32;     //~ ERROR suffixes on a tuple index are invalid
    let previous_carve_out_i32 = (42,).0u32;     //~ ERROR suffixes on a tuple index are invalid
    let previous_carve_out_isize = (42,).0isize; //~ ERROR suffixes on a tuple index are invalid
    let previous_carve_out_usize = (42,).0usize; //~ ERROR suffixes on a tuple index are invalid

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

fn numeral_struct_field_name_suffix_previous_carve_out() {
    let carve_out_struct_name = X { 0u32: 0, 1: 1, 2: 2 };
    //~^ ERROR suffixes on a tuple index are invalid
    match carve_out_struct_name {
        X { 0u32: _, .. } => {}
        //~^ ERROR suffixes on a tuple index are invalid
    }
}

// Unfortunately, it turns out `std::mem::offset_of!` uses the same expect suffix code path.
fn offset_of_suffix() {
    #[repr(C)]
    pub struct Struct<T>(u8, T);

    // Previous pseudo-FCW carve outs
    assert_eq!(std::mem::offset_of!(Struct<u32>, 0usize), 0);
    //~^ ERROR suffixes on a tuple index are invalid

    // Not part of carve outs
    assert_eq!(std::mem::offset_of!(Struct<u32>, 0u8), 0);
    //~^ ERROR suffixes on a tuple index are invalid
}
