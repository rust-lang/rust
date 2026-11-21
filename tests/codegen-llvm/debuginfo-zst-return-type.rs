//@ compile-flags: -C debuginfo=2 -Copt-level=0
//@ ignore-msvc the types are named differently

#![crate_type = "lib"]

pub struct Token;

pub enum SingleVariant {
    Only,
}

pub enum Uninhabited {}

// CHECK-DAG: !DISubprogram(name: "zst_struct", {{.*}}type: ![[TOKEN_FN:[0-9]+]], {{.*}}flags: DIFlagPrototyped,
// CHECK-DAG: ![[TOKEN_FN]] = !DISubroutineType(types: ![[TOKEN_TYS:[0-9]+]])
// CHECK-DAG: ![[TOKEN_TYS]] = !{![[TOKEN:[0-9]+]]}
// CHECK-DAG: ![[TOKEN]] = !DICompositeType(tag: DW_TAG_structure_type, name: "Token"
#[no_mangle]
pub fn zst_struct() -> Token {
    Token
}

// CHECK-DAG: !DISubprogram(name: "zst_enum", {{.*}}type: ![[ENUM_FN:[0-9]+]], {{.*}}flags: DIFlagPrototyped,
// CHECK-DAG: ![[ENUM_FN]] = !DISubroutineType(types: ![[ENUM_TYS:[0-9]+]])
// CHECK-DAG: ![[ENUM_TYS]] = !{![[ENUM:[0-9]+]]}
// CHECK-DAG: ![[ENUM]] = !DICompositeType(tag: DW_TAG_structure_type, name: "SingleVariant"
#[no_mangle]
pub fn zst_enum() -> SingleVariant {
    SingleVariant::Only
}

// CHECK-DAG: !DISubprogram(name: "uninhabited_enum", {{.*}}type: ![[UNINHAB_FN:[0-9]+]], {{.*}}flags: DIFlagPrototyped | DIFlagNoReturn,
// CHECK-DAG: ![[UNINHAB_FN]] = !DISubroutineType(types: ![[UNINHAB_TYS:[0-9]+]])
// CHECK-DAG: ![[UNINHAB_TYS]] = !{![[UNINHAB:[0-9]+]]}
// CHECK-DAG: ![[UNINHAB]] = !DICompositeType(tag: DW_TAG_structure_type, name: "Uninhabited"
#[no_mangle]
pub fn uninhabited_enum() -> Uninhabited {
    loop {}
}

// CHECK-DAG: !DISubprogram(name: "diverge", {{.*}}type: ![[NEVER_FN:[0-9]+]], {{.*}}flags: DIFlagPrototyped | DIFlagNoReturn,
// CHECK-DAG: ![[NEVER_FN]] = !DISubroutineType(types: ![[NEVER_TYS:[0-9]+]])
// CHECK-DAG: ![[NEVER_TYS]] = !{![[NEVER:[0-9]+]]}
// CHECK-DAG: ![[NEVER]] = !DIBasicType(name: "!"
#[no_mangle]
pub fn diverge() -> ! {
    loop {}
}

// CHECK-DAG: !DISubprogram(name: "unit", {{.*}}type: ![[UNIT_FN:[0-9]+]], {{.*}}flags: DIFlagPrototyped,
// CHECK-DAG: ![[UNIT_FN]] = !DISubroutineType(types: ![[UNIT_TYS:[0-9]+]])
// CHECK-DAG: ![[UNIT_TYS]] = !{null}
#[no_mangle]
pub fn unit() {}
