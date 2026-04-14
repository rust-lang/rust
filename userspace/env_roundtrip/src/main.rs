#![no_std]
#![no_main]
extern crate stem;

#[stem::main]
fn main() -> ! {
    let key = "THINGOS_TEST_VAR";
    let value = "roundtrip_ok";

    // set
    match stem::syscall::env_set(key.as_bytes(), value.as_bytes()) {
        Ok(_) => stem::println!("[env_roundtrip] set {}={:?}", key, value),
        Err(e) => {
            stem::println!("[env_roundtrip] FAIL: could not set var: {:?}", e);
            loop { stem::syscall::exit(1); }
        }
    }

    // get
    let mut buf = [0u8; 128];
    match stem::syscall::env_get(key.as_bytes(), &mut buf) {
        Ok(len) => {
            let v = core::str::from_utf8(&buf[..len]).unwrap_or("");
            if v == value {
                stem::println!("[env_roundtrip] get OK: {:?}", v);
            } else {
                stem::println!("[env_roundtrip] FAIL: expected {:?} got {:?}", value, v);
                loop { stem::syscall::exit(1); }
            }
        }
        Err(e) => {
            stem::println!("[env_roundtrip] FAIL: var not found: {:?}", e);
            loop { stem::syscall::exit(1); }
        }
    }

    // list
    let mut list_buf = [0u8; 1024];
    match stem::syscall::env_list(&mut list_buf) {
        Ok(_n) => {
            // env_list format: count:u32, then keylen:u32, key, vallen:u32, val
            let count = u32::from_le_bytes(list_buf[0..4].try_into().unwrap()) as usize;
            let mut found = false;
            let mut offset = 4;
            for _ in 0..count {
                let k_len = u32::from_le_bytes(list_buf[offset..offset+4].try_into().unwrap()) as usize;
                offset += 4;
                let k = core::str::from_utf8(&list_buf[offset..offset+k_len]).unwrap_or("");
                offset += k_len;
                let v_len = u32::from_le_bytes(list_buf[offset..offset+4].try_into().unwrap()) as usize;
                offset += 4;
                let v = core::str::from_utf8(&list_buf[offset..offset+v_len]).unwrap_or("");
                offset += v_len;

                if k == key && v == value {
                    found = true;
                    break;
                }
            }

            if found {
                stem::println!("[env_roundtrip] list OK: found key in {} vars", count);
            } else {
                stem::println!("[env_roundtrip] FAIL: key not found in vars() (total: {})", count);
                loop { stem::syscall::exit(1); }
            }
        }
        Err(e) => {
            stem::println!("[env_roundtrip] FAIL: could not list vars: {:?}", e);
            loop { stem::syscall::exit(1); }
        }
    }

    // unset
    match stem::syscall::env_unset(key.as_bytes()) {
        Ok(_) => stem::println!("[env_roundtrip] unset OK"),
        Err(e) => {
            stem::println!("[env_roundtrip] FAIL: could not unset var: {:?}", e);
            loop { stem::syscall::exit(1); }
        }
    }

    // verify unset
    match stem::syscall::env_get(key.as_bytes(), &mut buf) {
        Err(_) => stem::println!("[env_roundtrip] verify unset OK"),
        Ok(_) => {
            stem::println!("[env_roundtrip] FAIL: var still present after remove");
            loop { stem::syscall::exit(1); }
        }
    }

    stem::println!("[env_roundtrip] PASS");
    loop { stem::syscall::exit(0); }
}
