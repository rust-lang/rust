use core::panic::Location;

// Note: Some of the following tests depend on the source location,
// so please be careful when editing this file.

#[test]
fn location_const_caller() {
    const _CALLER_REFERENCE: &Location<'static> = Location::caller();
    const _CALLER: Location<'static> = *Location::caller();
}

#[test]
fn location_const_file() {
    const CALLER: &Location<'static> = Location::caller();
    const FILE: &str = CALLER.file();
    assert_eq!(FILE, file!());
}

#[test]
fn location_const_line() {
    const CALLER: &Location<'static> = Location::caller();
    const LINE: u32 = CALLER.line();
    assert_eq!(LINE, 21);
}

#[test]
fn location_const_column() {
    const CALLER: &Location<'static> = Location::caller();
    const COLUMN: u32 = CALLER.column();
    assert_eq!(COLUMN, 40);
}
