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
    assert!(matches!(abi, Err(AbiUnsupported::Unrecognized)))
}

#[test]
fn indices_are_correct() {
    for (i, abi_data) in AbiDatas.iter().enumerate() {
        assert_eq!(i, abi_data.abi.index());
    }
}
