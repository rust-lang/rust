//@ compile-flags: --crate-type=lib
//@ aux-build: unsafe-fields-crate-dep.rs

extern crate unsafe_fields_crate_dep;

use unsafe_fields_crate_dep::WithUnsafeField;

fn new_without_unsafe() -> WithUnsafeField {
    WithUnsafeField {
        //~^ ERROR
        unsafe_field: 0,
        safe_field: 0,
    }
}

fn operate_on_safe_field(s: &mut WithUnsafeField) {
    s.safe_field = 2;
    &s.safe_field;
    s.safe_field;
}

fn set_unsafe_field(s: &mut WithUnsafeField) {
    unsafe {
        s.unsafe_field = 2;
    }
}

fn read_unsafe_field(s: &WithUnsafeField) -> u32 {
    unsafe { s.unsafe_field }
}

fn ref_unsafe_field(s: &WithUnsafeField) -> &u32 {
    unsafe { &s.unsafe_field }
}

fn destructure(s: &WithUnsafeField) {
    unsafe {
        let WithUnsafeField { safe_field, unsafe_field } = s;
    }
}

fn set_unsafe_field_without_unsafe(s: &mut WithUnsafeField) {
    s.unsafe_field = 2;
    //~^ ERROR
}

fn read_unsafe_field_without_unsafe(s: &WithUnsafeField) -> u32 {
    s.unsafe_field
    //~^ ERROR
}

fn ref_unsafe_field_without_unsafe(s: &WithUnsafeField) -> &u32 {
    &s.unsafe_field
    //~^ ERROR
}

fn destructure_without_unsafe(s: &WithUnsafeField) {
    let WithUnsafeField { safe_field, unsafe_field } = s;
    //~^ ERROR

    let WithUnsafeField { safe_field, .. } = s;
}
