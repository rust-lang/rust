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
