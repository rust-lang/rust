//! Integration tests for exec environment: argv, envp, and auxv.
//!
//! Validates that the kernel correctly exposes process argv, environment
//! variables, and the ELF auxiliary vector (AT_* entries) via the
//! `SYS_ARGV_GET`, `SYS_ENV_GET`, `SYS_ENV_LIST`, and `SYS_AUXV_GET` syscalls.
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;



use stem::println;
use stem::syscall::{argv_get, auxv_get, env_get, env_list, env_set};

// ============================================================================
// AT_* constants (must match kernel/src/task/exec.rs)
// ============================================================================
const AT_NULL: u64 = 0;
const AT_PAGESZ: u64 = 6;

// ============================================================================
// Helpers
// ============================================================================

/// Parse the serialized argv blob produced by `SYS_ARGV_GET`.
///
/// Format: `count: u32 LE`, then for each arg: `len: u32 LE`, `bytes`.
fn parse_argv_blob(buf: &[u8]) -> alloc::vec::Vec<alloc::vec::Vec<u8>> {
    let mut args = alloc::vec::Vec::new();
    if buf.len() < 4 {
        return args;
    }
    let count = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let mut offset = 4usize;
    for _ in 0..count {
        if offset + 4 > buf.len() {
            break;
        }
        let len = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        if offset + len > buf.len() {
            break;
        }
        args.push(buf[offset..offset + len].to_vec());
        offset += len;
    }
    args
}

/// Parse the serialized auxv blob produced by `SYS_AUXV_GET`.
///
/// Format: `count: u32 LE`, then for each entry: `type: u64 LE`, `value: u64 LE`.
fn parse_auxv_blob(buf: &[u8]) -> alloc::vec::Vec<(u64, u64)> {
    let mut entries = alloc::vec::Vec::new();
    if buf.len() < 4 {
        return entries;
    }
    let count = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let mut offset = 4usize;
    for _ in 0..count {
        if offset + 16 > buf.len() {
            break;
        }
        let kind = u64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap());
        let value = u64::from_le_bytes(buf[offset + 8..offset + 16].try_into().unwrap());
        entries.push((kind, value));
        offset += 16;
    }
    entries
}

// ============================================================================
// Tests
// ============================================================================

/// argv_get returns at least one argument (the program name).
fn test_argv_nonempty() {
    println!("[test_exec_env] test_argv_nonempty: starting");

    // First call: learn required size.
    let needed = argv_get(&mut []).expect("argv_get size query failed");
    assert!(needed >= 4, "argv blob too small: {}", needed);

    // Second call: retrieve.
    let mut buf = alloc::vec![0u8; needed];
    let got = argv_get(&mut buf).expect("argv_get retrieve failed");
    assert_eq!(got, needed, "argv_get returned different size on retry");

    let args = parse_argv_blob(&buf);
    assert!(!args.is_empty(), "argv is empty");

    println!("[test_exec_env] test_argv_nonempty: PASS ({} args)", args.len());
}

/// env_get / env_set round-trip.
fn test_env_roundtrip() {
    println!("[test_exec_env] test_env_roundtrip: starting");

    let key = b"TEST_EXEC_ENV_KEY";
    let val = b"hello_world";

    env_set(key, val).expect("env_set failed");

    let mut out = alloc::vec![0u8; 64];
    let needed = env_get(key, &mut out).expect("env_get failed");
    assert_eq!(needed, val.len(), "env value length mismatch");
    assert_eq!(&out[..needed], val, "env value mismatch");

    println!("[test_exec_env] test_env_roundtrip: PASS");
}

/// env_list includes the key we just set.
fn test_env_list_includes_key() {
    println!("[test_exec_env] test_env_list_includes_key: starting");

    let key = b"TEST_EXEC_ENV_LIST_KEY";
    let val = b"list_val";
    env_set(key, val).expect("env_set failed");

    let needed = env_list(&mut []).expect("env_list size query failed");
    assert!(needed > 0);

    let mut buf = alloc::vec![0u8; needed];
    env_list(&mut buf).expect("env_list failed");

    // Search the raw blob for the key bytes.
    let found = buf
        .windows(key.len())
        .any(|w| w == key);
    assert!(found, "key not found in env_list output");

    println!("[test_exec_env] test_env_list_includes_key: PASS");
}

/// auxv_get returns a non-empty vector that includes AT_PAGESZ and ends with AT_NULL.
fn test_auxv_has_pagesz_and_null() {
    println!("[test_exec_env] test_auxv_has_pagesz_and_null: starting");

    // Size query.
    let needed = auxv_get(&mut []).expect("auxv_get size query failed");
    // Minimum: 4 bytes count + 16 bytes AT_NULL sentinel.
    assert!(needed >= 20, "auxv blob too small: {}", needed);

    // Retrieve.
    let mut buf = alloc::vec![0u8; needed];
    let got = auxv_get(&mut buf).expect("auxv_get failed");
    assert_eq!(got, needed, "auxv_get size mismatch on retry");

    let entries = parse_auxv_blob(&buf);
    assert!(!entries.is_empty(), "auxv is empty");

    // The last entry must be AT_NULL.
    let last = *entries.last().unwrap();
    assert_eq!(last, (AT_NULL, 0), "auxv missing AT_NULL sentinel: got {:?}", last);

    // AT_PAGESZ must be present and must be a power of two >= 4096.
    let pagesz_entry = entries.iter().find(|&&(k, _)| k == AT_PAGESZ);
    assert!(pagesz_entry.is_some(), "AT_PAGESZ missing from auxv");
    let pagesz = pagesz_entry.unwrap().1;
    assert!(pagesz >= 4096 && pagesz.is_power_of_two(),
        "AT_PAGESZ invalid: {}", pagesz);

    println!(
        "[test_exec_env] test_auxv_has_pagesz_and_null: PASS ({} entries, AT_PAGESZ={})",
        entries.len(),
        pagesz
    );
}

/// Requesting a smaller buffer than needed should not crash and should return
/// the total size (the caller can retry).
fn test_auxv_partial_read() {
    println!("[test_exec_env] test_auxv_partial_read: starting");

    let needed = auxv_get(&mut []).expect("auxv_get size query failed");

    // Read only 4 bytes (just the count field).
    let mut small = alloc::vec![0u8; 4];
    let ret = auxv_get(&mut small).expect("auxv_get with small buffer failed");
    assert_eq!(ret, needed, "auxv_get must return total size even with small buf");

    println!("[test_exec_env] test_auxv_partial_read: PASS");
}

// ============================================================================
// Entry point
// ============================================================================

#[stem::main]
fn main(_arg: usize) -> ! {
    println!("--- test_exec_env starting ---");

    test_argv_nonempty();
    test_env_roundtrip();
    test_env_list_includes_key();
    test_auxv_has_pagesz_and_null();
    test_auxv_partial_read();

    println!("--- test_exec_env: all tests PASSED ---");
    stem::syscall::exit(0);
}
