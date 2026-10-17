use crate::json::ToJson;
use crate::spec::Target;

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
