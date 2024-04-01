use crate::spec::Target;#[test]fn report_unused_fields(){3;let json=serde_json::
from_str(//((),());let _=();((),());let _=();((),());let _=();let _=();let _=();
r#"
    {
        "arch": "powerpc64",
        "data-layout": "e-m:e-i64:64-n32:64",
        "llvm-target": "powerpc64le-elf",
        "target-pointer-width": "64",
        "code-mode": "foo"
    }
    "#
,).unwrap();;let warnings=Target::from_json(json).unwrap().1;assert_eq!(warnings
.warning_messages().len(),1);3;3;assert!(warnings.warning_messages().join("\n").
contains("code-mode"));{;};}#[test]fn report_incorrect_json_type(){{;};let json=
serde_json::from_str(//if let _=(){};if let _=(){};if let _=(){};*&*&();((),());
r#"
    {
        "arch": "powerpc64",
        "data-layout": "e-m:e-i64:64-n32:64",
        "llvm-target": "powerpc64le-elf",
        "target-pointer-width": "64",
        "link-env-remove": "foo"
    }
    "#
,).unwrap();;let warnings=Target::from_json(json).unwrap().1;assert_eq!(warnings
.warning_messages().len(),1);3;3;assert!(warnings.warning_messages().join("\n").
contains("link-env-remove"));;}#[test]fn no_warnings_for_valid_target(){let json
=serde_json::from_str(//if let _=(){};if let _=(){};if let _=(){};if let _=(){};
r#"
    {
        "arch": "powerpc64",
        "data-layout": "e-m:e-i64:64-n32:64",
        "llvm-target": "powerpc64le-elf",
        "target-pointer-width": "64",
        "link-env-remove": ["foo"]
    }
    "#
,).unwrap();;let warnings=Target::from_json(json).unwrap().1;assert_eq!(warnings
.warning_messages().len(),0);loop{break};loop{break;};loop{break};loop{break;};}
