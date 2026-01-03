use rustc_public_bridge::IndexedVal;

use crate::abi::Layout;
use crate::mir::alloc::AllocId;
use crate::mir::mono::InstanceDef;
use crate::ty::{MirConstId, TyConstId, VariantIdx};
use crate::{CrateNum, DefId, Span, ThreadLocalIndex, Ty};

#[track_caller]
fn check_serialize<T: serde::Serialize>(value: T, expected_json: &str) {
    let got_json = serde_json::to_string(&value).unwrap();
    assert_eq!(got_json, expected_json, "didn't get expected json for serializing");
}

#[test]
fn serialize_cratenum() {
    check_serialize(CrateNum(1, ThreadLocalIndex), "1");
}

#[test]
fn serialize_defid() {
    check_serialize(DefId::to_val(2), "2");
}

#[test]
fn serialize_layout() {
    check_serialize(Layout::to_val(3), "3");
}

#[test]
fn serialize_allocid() {
    check_serialize(AllocId::to_val(4), "4");
}

#[test]
fn serialize_ty() {
    check_serialize(Ty::to_val(5), "5");
}

#[test]
fn serialize_tyconstid() {
    check_serialize(TyConstId::to_val(6), "6");
}

#[test]
fn serialize_mirconstid() {
    check_serialize(MirConstId::to_val(7), "7");
}

#[test]
fn serialize_span() {
    check_serialize(Span::to_val(8), "8");
}

#[test]
fn serialize_variantidx() {
    check_serialize(VariantIdx::to_val(9), "9");
}

#[test]
fn serialize_instancedef() {
    check_serialize(InstanceDef::to_val(10), "10");
}
