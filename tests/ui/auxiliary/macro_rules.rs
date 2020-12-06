#![allow(dead_code)]

//! Used to test that certain lints don't trigger in imported external macros

#[macro_export]
macro_rules! foofoo {
    () => {
        loop {}
    };
}

#[macro_export]
macro_rules! must_use_unit {
    () => {
        #[must_use]
        fn foo() {}
    };
}

#[macro_export]
macro_rules! try_err {
    () => {
        pub fn try_err_fn() -> Result<i32, i32> {
            let err: i32 = 1;
            // To avoid warnings during rustfix
            if true {
                Err(err)?
            } else {
                Ok(2)
            }
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
macro_rules! take_external {
    ($s:expr) => {
        std::mem::replace($s, Default::default())
    };
}

#[macro_export]
macro_rules! option_env_unwrap_external {
    ($env: expr) => {
        option_env!($env).unwrap()
    };
    ($env: expr, $message: expr) => {
        option_env!($env).expect($message)
    };
}

#[macro_export]
macro_rules! ref_arg_binding {
    () => {
        let ref _y = 42;
    };
}

#[macro_export]
macro_rules! ref_arg_function {
    () => {
        fn fun_example(ref _x: usize) {}
    };
}

#[macro_export]
macro_rules! as_conv_with_arg {
    (0u32 as u64) => {
        ()
    };
}

#[macro_export]
macro_rules! as_conv {
    () => {
        0u32 as u64
    };
}
