#![crate_type = "staticlib"]

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[no_mangle]
pub extern "C" fn my_add(a: i32, b: i32) -> i32 {
    a + b
}

// A leading `\x01` suppresses decoration; the object file contains the name verbatim.
#[used]
#[export_name = "\x01my_no_mangle_export"]
pub static MY_NO_MANGLE_EXPORT: [u8; 4] = *b"keep";

#[no_mangle]
pub extern "C" fn my_hash_lookup(key: u64) -> u64 {
    let mut map = HashMap::new();
    for i in 0..100u64 {
        map.insert(i, i.wrapping_mul(2654435761));
    }
    *map.get(&key).unwrap_or(&0)
}

fn internal_helper() -> i32 {
    42
}

#[no_mangle]
pub extern "C" fn call_internal() -> i32 {
    internal_helper()
}

#[no_mangle]
pub extern "C" fn my_safe_div(a: i32, b: i32) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        if b == 0 {
            panic!("division by zero!");
        }
        a / b
    })) {
        Ok(result) => result,
        Err(_) => -1,
    }
}
