// Test that Objective-C class and selector references inlined across crates
// get defined in this CGU but non-inline references don't.

// ignore-tidy-linelength
//@ aux-build: darwin_objc_aux.rs
//@ revisions: x86_64_macos aarch64_macos
//@ [x86_64_macos] only-x86_64-apple-darwin
//@ [aarch64_macos] only-aarch64-apple-darwin

#![crate_type = "lib"]
#![feature(darwin_objc)]

use std::os::darwin::objc;

extern crate darwin_objc_aux as aux;

#[no_mangle]
pub fn get_object_class() -> objc::Class {
    aux::inline_get_object_class()
}

#[no_mangle]
pub fn get_alloc_selector() -> objc::SEL {
    aux::inline_get_alloc_selector()
}

#[no_mangle]
pub fn get_string_class() -> objc::Class {
    aux::never_inline_get_string_class()
}

#[no_mangle]
pub fn get_init_selector() -> objc::SEL {
    aux::never_inline_get_init_selector()
}

// CHECK: %struct._class_t = type { ptr, ptr, ptr, ptr, ptr }

// CHECK: @"OBJC_CLASS_$_NSObject" = external global %struct._class_t
// CHECK: @"OBJC_CLASSLIST_REFERENCES_$_.{{[0-9]+}}" = internal global ptr @"OBJC_CLASS_$_NSObject", section "__DATA,__objc_classrefs,regular,no_dead_strip", align 8

// CHECK: @OBJC_METH_VAR_NAME_.{{[0-9]+}} = private unnamed_addr constant [6 x i8] c"alloc\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK: @OBJC_SELECTOR_REFERENCES_.{{[0-9]+}} = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_.{{[0-9]+}}, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8

// CHECK-NOT: @"OBJC_CLASS_$_NSString" = external global %struct._class_t
// CHECK-NOT: @"OBJC_CLASSLIST_REFERENCES_$_.{{[0-9]+}}" = internal global ptr @"OBJC_CLASS_$_NSString"

// CHECK-NOT: @OBJC_METH_VAR_NAME_.{{[0-9]+}} = private unnamed_addr constant [5 x i8] c"init\00"
// CHECK-NOT: @OBJC_SELECTOR_REFERENCES_.{{[0-9]+}} = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_.{{[0-9]+}}

// CHECK: load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.{{[0-9]+}}", align 8
// CHECK: load ptr, ptr @OBJC_SELECTOR_REFERENCES_.{{[0-9]+}}, align 8

// CHECK: !{{[0-9]+}} = !{i32 1, !"Objective-C Version", i32 2}
// CHECK: !{{[0-9]+}} = !{i32 1, !"Objective-C Image Info Version", i32 0}
// CHECK: !{{[0-9]+}} = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
// CHECK-NOT: !{{[0-9]+}} = !{i32 1, !"Objective-C Is Simulated", i32 32}
// CHECK: !{{[0-9]+}} = !{i32 1, !"Objective-C Class Properties", i32 64}
