//@ run-pass
//@ aux-build:attributes_aux.rs

#![feature(type_info)]
#![feature(register_tool)]
#![register_tool(my_tool)]
#![allow(dead_code)]

extern crate attributes_aux;

use std::mem::type_info::{Type, TypeKind};

#[allow(dead_code)]
#[doc = "A test struct"]
struct AttrStruct {
    #[allow(unused)]
    first: u8,
    second: u16,
}

#[rustfmt::skip]
struct ToolAttr {
    x: u8,
}

#[my_tool::category = "test_value"]
struct EqAttr {
    x: u8,
}

#[repr(C)]
#[allow(dead_code)]
struct ReprCStruct {
    x: u32,
    y: u8,
}

fn main() {
    // #[doc = "..."] is a parsed built-in attribute, not reflected in `attributes`.
    let Type { kind: TypeKind::Struct(ty), .. } = (const { Type::of::<AttrStruct>() }) else {
        panic!()
    };
    assert_eq!(ty.attributes.len(), 1); // only `allow`, not `doc`
    assert_eq!(ty.attributes[0].path, "allow");
    assert_eq!(ty.attributes[0].args, "dead_code");
    assert_eq!(ty.fields[0].attributes.len(), 1);
    assert_eq!(ty.fields[0].attributes[0].path, "allow");

    let Type { kind: TypeKind::Struct(ty), .. } = (const { Type::of::<ToolAttr>() }) else {
        panic!()
    };
    assert_eq!(ty.attributes.len(), 1);
    assert_eq!(ty.attributes[0].path, "rustfmt::skip");
    assert_eq!(ty.attributes[0].args, "");

    let Type { kind: TypeKind::Struct(ty), .. } = (const { Type::of::<EqAttr>() }) else {
        panic!()
    };
    assert_eq!(ty.attributes.len(), 1);
    assert_eq!(ty.attributes[0].path, "my_tool::category");
    assert_eq!(ty.attributes[0].args, "test_value");

    // #[repr(C)] is a parsed built-in attribute, not reflected in `attributes`.
    let Type { kind: TypeKind::Struct(ty), .. } = (const { Type::of::<ReprCStruct>() }) else {
        panic!()
    };
    assert_eq!(ty.attributes.len(), 1); // only `allow`, not `repr`
    assert_eq!(ty.attributes[0].path, "allow");
    assert_eq!(ty.attributes[0].args, "dead_code");

    // Cross-crate: only tool/custom attributes survive metadata serialization.
    let Type { kind: TypeKind::Enum(ty), .. } =
        (const { Type::of::<attributes_aux::CrossCrateEnum>() })
    else {
        panic!()
    };

    assert!(ty.non_exhaustive);
    assert_eq!(ty.attributes.len(), 1);
    assert_eq!(ty.attributes[0].path, "my_tool::enum_tag");
    assert_eq!(ty.attributes[0].args, "cross_crate");

    assert_eq!(ty.variants.len(), 2);
    assert_eq!(ty.variants[0].attributes.len(), 1);
    assert_eq!(ty.variants[0].attributes[0].path, "my_tool::variant_tag");
    assert_eq!(ty.variants[0].attributes[0].args, "variant_a");
    assert_eq!(ty.variants[0].fields.len(), 1);
    assert_eq!(ty.variants[0].fields[0].attributes.len(), 1);
    assert_eq!(ty.variants[0].fields[0].attributes[0].path, "my_tool::field_tag");
    assert_eq!(ty.variants[0].fields[0].attributes[0].args, "variant_field");

    assert_eq!(ty.variants[1].attributes.len(), 1);
    assert_eq!(ty.variants[1].attributes[0].path, "my_tool::empty_variant_tag");
    assert_eq!(ty.variants[1].attributes[0].args, "");

    let Type { kind: TypeKind::Struct(ty), .. } =
        (const { Type::of::<attributes_aux::CrossCrateStruct>() })
    else {
        panic!()
    };

    assert!(ty.non_exhaustive);
    // Same as above: only tool/custom attributes are visible cross-crate.
    assert_eq!(ty.attributes.len(), 1);
    assert_eq!(ty.attributes[0].path, "my_tool::struct_tag");
    assert_eq!(ty.attributes[0].args, "cross_crate_struct");
    assert_eq!(ty.fields.len(), 2);
    assert_eq!(ty.fields[0].attributes.len(), 1);
    assert_eq!(ty.fields[0].attributes[0].path, "my_tool::field_tag");
    assert_eq!(ty.fields[0].attributes[0].args, "public_field");
    assert!(ty.fields[1].attributes.is_empty());
}
