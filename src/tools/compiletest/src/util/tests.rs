use super::*;

#[test]
#[should_panic(expected = "Cannot determine Architecture from triple")]
fn test_get_arch_failure() {
    get_arch("abc");
}

#[test]
fn test_get_arch() {
    assert_eq!("x86_64", get_arch("x86_64-unknown-linux-gnu"));
    assert_eq!("x86_64", get_arch("amd64"));
    assert_eq!("nvptx64", get_arch("nvptx64-nvidia-cuda"));
}

#[test]
#[should_panic(expected = "Cannot determine OS from triple")]
fn test_matches_os_failure() {
    matches_os("abc", "abc");
}

#[test]
fn test_matches_os() {
    assert!(matches_os("x86_64-unknown-linux-gnu", "linux"));
    assert!(matches_os("wasm32-unknown-unknown", "emscripten"));
    assert!(matches_os("wasm32-unknown-unknown", "wasm32-bare"));
    assert!(!matches_os("wasm32-unknown-unknown", "windows"));
    assert!(matches_os("thumbv6m0-none-eabi", "none"));
    assert!(matches_os("riscv32imc-unknown-none-elf", "none"));
    assert!(matches_os("nvptx64-nvidia-cuda", "cuda"));
    assert!(matches_os("x86_64-fortanix-unknown-sgx", "sgx"));
}

#[test]
fn is_big_endian_test() {
    assert!(!is_big_endian("no"));
    assert!(is_big_endian("sparc-unknown-unknown"));
}

#[test]
fn path_buf_with_extra_extension_test() {
    assert_eq!(
        PathBuf::from("foo.rs.stderr"),
        PathBuf::from("foo.rs").with_extra_extension("stderr")
    );
    assert_eq!(
        PathBuf::from("foo.rs.stderr"),
        PathBuf::from("foo.rs").with_extra_extension(".stderr")
    );
    assert_eq!(PathBuf::from("foo.rs"), PathBuf::from("foo.rs").with_extra_extension(""));
}
