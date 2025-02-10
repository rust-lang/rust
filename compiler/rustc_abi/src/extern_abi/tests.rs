use std::assert_matches::assert_matches;

use super::*;

#[allow(non_snake_case)]
#[test]
fn lookup_Rust() {
    let abi = lookup("Rust");
    assert!(abi.is_ok() && abi.unwrap().data().name == "Rust");
}

#[test]
fn lookup_cdecl() {
    let abi = lookup("cdecl");
    assert!(abi.is_ok() && abi.unwrap().data().name == "cdecl");
}

#[test]
fn lookup_baz() {
    let abi = lookup("baz");
    assert_matches!(abi, Err(AbiUnsupported {}));
}

#[test]
fn indices_are_correct() {
    for (i, abi_data) in AbiDatas.iter().enumerate() {
        assert_eq!(i, abi_data.abi.index());
    }
}

#[test]
fn guarantee_lexicographic_ordering() {
    let abis = ExternAbi::ALL_VARIANTS;
    let mut sorted_abis = abis.to_vec();
    sorted_abis.sort_unstable();
    assert_eq!(abis, sorted_abis);
}
