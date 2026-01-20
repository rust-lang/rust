use std::str::FromStr;

use rustc_data_structures::assert_matches;

use super::*;

#[allow(non_snake_case)]
#[test]
fn lookup_Rust() {
    let abi = ExternAbi::from_str("Rust");
    assert!(abi.is_ok() && abi.unwrap().as_str() == "Rust");
}

#[test]
fn lookup_cdecl() {
    let abi = ExternAbi::from_str("cdecl");
    assert!(abi.is_ok() && abi.unwrap().as_str() == "cdecl");
}

#[test]
fn lookup_baz() {
    let abi = ExternAbi::from_str("baz");
    assert_matches!(abi, Err(AbiFromStrErr::Unknown));
}

#[test]
fn guarantee_lexicographic_ordering() {
    let abis = ExternAbi::ALL_VARIANTS;
    let mut sorted_abis = abis.to_vec();
    sorted_abis.sort_unstable();
    assert_eq!(abis, sorted_abis);
}
