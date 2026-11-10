use std::borrow::Cow;
use std::collections::BTreeMap;

use crate::json::ToJson;
use crate::spec::{Target, TargetTuple};

#[test]
fn report_unused_fields() {
    let json = r#"
    {
        "arch": "powerpc64",
        "data-layout": "e-m:e-i64:64-n32:64",
        "llvm-target": "powerpc64le-elf",
        "target-pointer-width": 64,
        "code-mode": "foo"
    }
    "#;
    let result = Target::from_json(json);
    eprintln!("{result:#?}");
    assert!(result.is_err());
}

#[test]
fn custom_arch_propagates_from_json() {
    let json = r#"
    {
        "llvm-target": "x86_64-unknown-none-gnu",
        "data-layout": "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
        "arch": "customarch",
        "target-endian": "little",
        "target-pointer-width": 64,
        "os": "customos",
        "linker-flavor": "ld.lld",
        "linker": "rust-lld",
        "executables": true
    }
    "#;
    rustc_span::create_session_if_not_set_then(rustc_span::edition::DEFAULT_EDITION, |_| {
        let (target, warnings) = Target::from_json(json).expect("json target parses");
        assert!(warnings.warning_messages().is_empty());
        assert_eq!(target.arch.desc(), "customarch");
        let serialized = target.to_json();
        assert_eq!(serialized["arch"], "customarch");
    });
}

fn bpf_target_with_versioned_args(map: BTreeMap<u32, Cow<'static, [Cow<'static, str>]>>) -> Target {
    let mut t = Target::expect_builtin(&TargetTuple::from_tuple("bpfeb-unknown-none"));
    t.options.llvm_args_by_version = map;
    t
}

#[test]
fn llvm_args_by_version_empty_map_returns_no_args() {
    let t = bpf_target_with_versioned_args(BTreeMap::new());
    assert!(t.llvm_args_for_llvm_version(1).collect::<Vec<_>>().is_empty());
    assert!(t.llvm_args_for_llvm_version(9999).collect::<Vec<_>>().is_empty());
}

#[test]
fn llvm_args_by_version_excludes_entries_above_major() {
    let t = bpf_target_with_versioned_args(BTreeMap::from([
        (23u32, Cow::Borrowed(&[Cow::Borrowed("-a")][..])),
        (24, Cow::Borrowed(&[Cow::Borrowed("-b")][..])),
    ]));
    assert!(t.llvm_args_for_llvm_version(22).collect::<Vec<_>>().is_empty());
    assert_eq!(t.llvm_args_for_llvm_version(23).collect::<Vec<_>>(), &[&Cow::Borrowed("-a")]);
    assert_eq!(
        t.llvm_args_for_llvm_version(24).collect::<Vec<_>>(),
        &[&Cow::Borrowed("-a"), &Cow::Borrowed("-b")]
    );
    assert_eq!(
        t.llvm_args_for_llvm_version(25).collect::<Vec<_>>(),
        &[&Cow::Borrowed("-a"), &Cow::Borrowed("-b")]
    );
}

#[test]
fn llvm_args_by_version_accumulates_multiple_entries() {
    let t = bpf_target_with_versioned_args(BTreeMap::from([
        (23u32, Cow::Borrowed(&[Cow::Borrowed("-a"), Cow::Borrowed("-b")][..])),
        (24, Cow::Borrowed(&[Cow::Borrowed("-c")][..])),
    ]));
    assert_eq!(
        t.llvm_args_for_llvm_version(23).collect::<Vec<_>>(),
        &[&Cow::Borrowed("-a"), &Cow::Borrowed("-b")]
    );
    assert_eq!(
        t.llvm_args_for_llvm_version(24).collect::<Vec<_>>(),
        &[&Cow::Borrowed("-a"), &Cow::Borrowed("-b"), &Cow::Borrowed("-c")]
    );
    assert_eq!(
        t.llvm_args_for_llvm_version(100).collect::<Vec<_>>(),
        &[&Cow::Borrowed("-a"), &Cow::Borrowed("-b"), &Cow::Borrowed("-c")]
    );
}
