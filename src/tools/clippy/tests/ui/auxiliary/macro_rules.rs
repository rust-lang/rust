#![allow(dead_code)]

//! Used to test that certain lints don't trigger in imported external macros
#[macro_export]
macro_rules! try_err {
    () => {
        pub fn try_err_fn() -> Result<i32, i32> {
            let err: i32 = 1;
            // To avoid warnings during rustfix
            if true { Err(err)? } else { Ok(2) }
        }
    };
}

#[macro_export]
macro_rules! string_add {
    () => {
        let y = "".to_owned();
        let z = y + "...";
    };
}

#[macro_export]
macro_rules! string_lit_as_bytes {
    ($s:literal) => {
        const C: &[u8] = $s.as_bytes();
    };
}

#[macro_export]
macro_rules! mut_mut {
    () => {
        let mut_mut_ty: &mut &mut u32 = &mut &mut 1u32;
    };
}

#[macro_export]
macro_rules! issue_10421 {
    () => {
        let mut a = 1;
        let mut b = 2;
        a = b;
        b = a;
    };
}

#[macro_export]
macro_rules! macro_with_panic {
    () => {
        panic!()
    };
}

#[macro_export]
macro_rules! bad_transmute {
    ($e:expr) => {
        std::mem::transmute($e)
    };
}
