#![warn(clippy::const_is_empty)]
#![allow(clippy::needless_late_init, unused_must_use)]

fn test_literal() {
    if "".is_empty() {
        //~^ const_is_empty
    }
    if "foobar".is_empty() {
        //~^ const_is_empty
    }
}

fn test_byte_literal() {
    if b"".is_empty() {
        //~^ const_is_empty
    }
    if b"foobar".is_empty() {
        //~^ const_is_empty
    }
}

fn test_no_mut() {
    let mut empty = "";
    if empty.is_empty() {
        // No lint because it is mutable
    }
}

fn test_propagated() {
    let empty = "";
    let non_empty = "foobar";
    let empty2 = empty;
    let non_empty2 = non_empty;
    if empty2.is_empty() {
        //~^ const_is_empty
    }
    if non_empty2.is_empty() {
        //~^ const_is_empty
    }
}

const EMPTY_STR: &str = "";
const NON_EMPTY_STR: &str = "foo";
const EMPTY_BSTR: &[u8] = b"";
const NON_EMPTY_BSTR: &[u8] = b"foo";
const EMPTY_U8_SLICE: &[u8] = &[];
const NON_EMPTY_U8_SLICE: &[u8] = &[1, 2];
const EMPTY_SLICE: &[u32] = &[];
const NON_EMPTY_SLICE: &[u32] = &[1, 2];
const NON_EMPTY_SLICE_REPEAT: &[u32] = &[1; 2];
const EMPTY_ARRAY: [u32; 0] = [];
const EMPTY_ARRAY_REPEAT: [u32; 0] = [1; 0];
const NON_EMPTY_ARRAY: [u32; 2] = [1, 2];
const NON_EMPTY_ARRAY_REPEAT: [u32; 2] = [1; 2];
const EMPTY_REF_ARRAY: &[u32; 0] = &[];
const NON_EMPTY_REF_ARRAY: &[u32; 3] = &[1, 2, 3];

fn test_from_const() {
    let _ = EMPTY_STR.is_empty();
    //~^ const_is_empty

    let _ = NON_EMPTY_STR.is_empty();
    //~^ const_is_empty

    let _ = EMPTY_BSTR.is_empty();
    //~^ const_is_empty

    let _ = NON_EMPTY_BSTR.is_empty();
    //~^ const_is_empty

    let _ = EMPTY_ARRAY.is_empty();
    //~^ const_is_empty

    let _ = EMPTY_ARRAY_REPEAT.is_empty();
    //~^ const_is_empty

    let _ = EMPTY_U8_SLICE.is_empty();
    //~^ const_is_empty

    let _ = NON_EMPTY_U8_SLICE.is_empty();
    //~^ const_is_empty

    let _ = NON_EMPTY_ARRAY.is_empty();
    //~^ const_is_empty

    let _ = NON_EMPTY_ARRAY_REPEAT.is_empty();
    //~^ const_is_empty

    let _ = EMPTY_REF_ARRAY.is_empty();
    //~^ const_is_empty

    let _ = NON_EMPTY_REF_ARRAY.is_empty();
    //~^ const_is_empty

    let _ = EMPTY_SLICE.is_empty();
    //~^ const_is_empty

    let _ = NON_EMPTY_SLICE.is_empty();
    //~^ const_is_empty

    let _ = NON_EMPTY_SLICE_REPEAT.is_empty();
    //~^ const_is_empty
}

fn main() {
    let value = "foobar";
    let _ = value.is_empty();
    //~^ const_is_empty

    let x = value;
    let _ = x.is_empty();
    //~^ const_is_empty

    let _ = "".is_empty();
    //~^ const_is_empty

    let _ = b"".is_empty();
    //~^ const_is_empty
}

fn str_from_arg(var: &str) {
    var.is_empty();
    // Do not lint, we know nothiny about var
}

fn update_str() {
    let mut value = "duck";
    value = "penguin";

    let _ = value.is_empty();
    // Do not lint since value is mutable
}

fn macros() {
    // Content from Macro
    let file = include_str!("const_is_empty.rs");
    let _ = file.is_empty();
    // No lint because initializer comes from a macro result

    let var = env!("PATH");
    let _ = var.is_empty();
    // No lint because initializer comes from a macro result
}

fn conditional_value() {
    let value;

    if true {
        value = "hey";
    } else {
        value = "hej";
    }

    let _ = value.is_empty();
    // Do not lint, current constant folding is too simple to detect this
}

fn cfg_conditioned() {
    #[cfg(test)]
    let val = "";
    #[cfg(not(test))]
    let val = "foo";

    let _ = val.is_empty();
    // Do not lint, value depend on a #[cfg(…)] directive
}

fn not_cfg_conditioned() {
    let val = "";
    #[cfg(not(target_os = "inexistent"))]
    let _ = val.is_empty();
    //~^ const_is_empty
}

const fn const_rand() -> &'static str {
    "17"
}

fn const_expressions() {
    let _ = const { if true { "1" } else { "2" } }.is_empty();
    // Do not lint, we do not recurse into boolean expressions

    let _ = const_rand().is_empty();
    // Do not lint, we do not recurse into functions
}

fn constant_from_external_crate() {
    let _ = std::env::consts::EXE_EXTENSION.is_empty();
    // Do not lint, `exe_ext` comes from the `std` crate
}

fn issue_13106() {
    const {
        assert!(!NON_EMPTY_STR.is_empty());
    }

    const {
        assert!(EMPTY_STR.is_empty());
    }

    const {
        EMPTY_STR.is_empty();
        //~^ const_is_empty
    }
}
