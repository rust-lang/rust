//@ compile-flags: -C debuginfo=2 -C opt-level=0
#![crate_type = "lib"]
#![feature(btf_tags)]

#[btf_decl_tag("g_tag")]
#[btf_type_tag("g_ptr")]
#[no_mangle]
pub static mut G: *mut u32 = core::ptr::null_mut();

#[btf_decl_tag("s_tag")]
pub struct S {
    #[btf_decl_tag("f_tag")]
    pub a: u32,
    #[btf_type_tag("user")]
    #[btf_type_tag("rcu")]
    pub p: *const u64,
    pub plain: *const u64,
}

#[btf_decl_tag("u_tag")]
pub union U {
    #[btf_decl_tag("uf_tag")]
    pub x: u32,
    pub y: f32,
}

#[btf_decl_tag("fn_tag")]
#[no_mangle]
pub fn func(
    #[btf_decl_tag("arg_tag")]
    #[btf_type_tag("user")]
    buf: *mut u8,
    s: &S,
    u: U,
) -> u32 {
    unsafe { *buf as u32 + s.a + u.x }
}

#[no_mangle]
pub fn untagged(buf: *mut u8) -> u8 {
    unsafe { *buf }
}

// CHECK-DAG: !DIGlobalVariable(name: "G",{{.*}} type: ![[G_TY:[0-9]+]],{{.*}} annotations: ![[G_ANN:[0-9]+]])
// CHECK-DAG: ![[G_ANN]] = !{![[G_TAG:[0-9]+]]}
// CHECK-DAG: ![[G_TAG]] = !{!"btf_decl_tag", !"g_tag"}
// CHECK-DAG: ![[G_TY]] = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*mut u32",{{.*}} annotations: ![[G_PTR_ANN:[0-9]+]])
// CHECK-DAG: ![[G_PTR_ANN]] = !{![[G_PTR_TAG:[0-9]+]]}
// CHECK-DAG: ![[G_PTR_TAG]] = !{!"btf_type_tag", !"g_ptr"}

// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "S",{{.*}} annotations: ![[S_ANN:[0-9]+]])
// CHECK-DAG: ![[S_ANN]] = !{![[S_TAG:[0-9]+]]}
// CHECK-DAG: ![[S_TAG]] = !{!"btf_decl_tag", !"s_tag"}
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "a",{{.*}} annotations: ![[A_ANN:[0-9]+]])
// CHECK-DAG: ![[A_ANN]] = !{![[A_TAG:[0-9]+]]}
// CHECK-DAG: ![[A_TAG]] = !{!"btf_decl_tag", !"f_tag"}
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "p",{{.*}} baseType: ![[P_TY:[0-9]+]],
// CHECK-DAG: ![[P_TY]] = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const u64",{{.*}} annotations: ![[P_ANN:[0-9]+]])
// CHECK-DAG: ![[P_ANN]] = !{![[USER:[0-9]+]], ![[RCU:[0-9]+]]}
// CHECK-DAG: ![[USER]] = !{!"btf_type_tag", !"user"}
// CHECK-DAG: ![[RCU]] = !{!"btf_type_tag", !"rcu"}
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "plain",{{.*}} baseType: ![[PLAIN_TY:[0-9]+]],
// CHECK-DAG: ![[PLAIN_TY]] = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const u64",{{.*}} align: {{[0-9]+}}{{(, dwarfAddressSpace: 0)?}})

// CHECK-DAG: !DICompositeType(tag: DW_TAG_union_type, name: "U",{{.*}} annotations: ![[U_ANN:[0-9]+]])
// CHECK-DAG: ![[U_ANN]] = !{![[U_TAG:[0-9]+]]}
// CHECK-DAG: ![[U_TAG]] = !{!"btf_decl_tag", !"u_tag"}
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "x",{{.*}} annotations: ![[X_ANN:[0-9]+]])
// CHECK-DAG: ![[X_ANN]] = !{![[X_TAG:[0-9]+]]}
// CHECK-DAG: ![[X_TAG]] = !{!"btf_decl_tag", !"uf_tag"}

// CHECK-DAG: distinct !DISubprogram(name: "func",{{.*}} type: ![[FUNC_TY:[0-9]+]],{{.*}} annotations: ![[FN_ANN:[0-9]+]])
// CHECK-DAG: ![[FN_ANN]] = !{![[FN_TAG:[0-9]+]]}
// CHECK-DAG: ![[FN_TAG]] = !{!"btf_decl_tag", !"fn_tag"}
// CHECK-DAG: ![[FUNC_TY]] = !DISubroutineType(types: ![[FUNC_TYS:[0-9]+]])
// CHECK-DAG: ![[FUNC_TYS]] = !{!{{[0-9]+}}, ![[BUF_TY:[0-9]+]], !{{[0-9]+}}, !{{[0-9]+}}}
// CHECK-DAG: ![[BUF_TY]] = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*mut u8",{{.*}} annotations: ![[BUF_TY_ANN:[0-9]+]])
// CHECK-DAG: ![[BUF_TY_ANN]] = !{![[USER]]}
// CHECK-DAG: !DILocalVariable(name: "buf", arg: 1,{{.*}} type: ![[BUF_TY]],{{.*}} annotations: ![[ARG_ANN:[0-9]+]])
// CHECK-DAG: ![[ARG_ANN]] = !{![[ARG_TAG:[0-9]+]]}
// CHECK-DAG: ![[ARG_TAG]] = !{!"btf_decl_tag", !"arg_tag"}

// Tags belong to the declaration, so they must not leak into other uses of the same type.
// CHECK-DAG: distinct !DISubprogram(name: "untagged",{{.*}} type: ![[UNTAGGED_TY:[0-9]+]],{{.*}} retainedNodes:
// CHECK-DAG: ![[UNTAGGED_TY]] = !DISubroutineType(types: ![[UNTAGGED_TYS:[0-9]+]])
// CHECK-DAG: ![[UNTAGGED_TYS]] = !{!{{[0-9]+}}, ![[PLAIN_BUF_TY:[0-9]+]]}
// CHECK-DAG: ![[PLAIN_BUF_TY]] = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*mut u8",{{.*}} align: {{[0-9]+}}{{(, dwarfAddressSpace: 0)?}})
