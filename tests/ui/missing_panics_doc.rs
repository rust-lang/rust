#![warn(clippy::missing_panics_doc)]
#![allow(clippy::option_map_unit_fn)]
fn main() {}

/// This needs to be documented
pub fn unwrap() {
    let result = Err("Hi");
    result.unwrap()
}

/// This needs to be documented
pub fn panic() {
    panic!("This function panics")
}

/// This needs to be documented
pub fn todo() {
    todo!()
}

/// This needs to be documented
pub fn inner_body(opt: Option<u32>) {
    opt.map(|x| {
        if x == 10 {
            panic!()
        }
    });
}

/// This needs to be documented
pub fn unreachable_and_panic() {
    if true { unreachable!() } else { panic!() }
}

/// This needs to be documented
pub fn assert_eq() {
    let x = 0;
    assert_eq!(x, 0);
}

/// This needs to be documented
pub fn assert_ne() {
    let x = 0;
    assert_ne!(x, 0);
}

/// This is documented
///
/// # Panics
///
/// Panics if `result` if an error
pub fn unwrap_documented() {
    let result = Err("Hi");
    result.unwrap()
}

/// This is documented
///
/// # Panics
///
/// Panics just because
pub fn panic_documented() {
    panic!("This function panics")
}

/// This is documented
///
/// # Panics
///
/// Panics if `opt` is Just(10)
pub fn inner_body_documented(opt: Option<u32>) {
    opt.map(|x| {
        if x == 10 {
            panic!()
        }
    });
}

/// This is documented
///
/// # Panics
///
/// We still need to do this part
pub fn todo_documented() {
    todo!()
}

/// This is documented
///
/// # Panics
///
/// We still need to do this part
pub fn unreachable_amd_panic_documented() {
    if true { unreachable!() } else { panic!() }
}

/// This is documented
///
/// # Panics
///
/// Panics if `x` is not 0.
pub fn assert_eq_documented() {
    let x = 0;
    assert_eq!(x, 0);
}

/// This is documented
///
/// # Panics
///
/// Panics if `x` is 0.
pub fn assert_ne_documented() {
    let x = 0;
    assert_ne!(x, 0);
}

/// This is okay because it is private
fn unwrap_private() {
    let result = Err("Hi");
    result.unwrap()
}

/// This is okay because it is private
fn panic_private() {
    panic!("This function panics")
}

/// This is okay because it is private
fn todo_private() {
    todo!()
}

/// This is okay because it is private
fn inner_body_private(opt: Option<u32>) {
    opt.map(|x| {
        if x == 10 {
            panic!()
        }
    });
}

/// This is okay because unreachable
pub fn unreachable() {
    unreachable!("This function panics")
}

/// #6970.
/// This is okay because it is expansion of `debug_assert` family.
pub fn debug_assertions() {
    debug_assert!(false);
    debug_assert_eq!(1, 2);
    debug_assert_ne!(1, 2);
}
