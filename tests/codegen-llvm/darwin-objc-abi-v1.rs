// ignore-tidy-linelength
//@ add-core-stubs
//@ revisions: i686_apple_darwin
//@ [i686_apple_darwin] compile-flags: --target i686-apple-darwin
//@ [i686_apple_darwin] needs-llvm-components: x86

#![crate_type = "lib"]
#![feature(no_core, lang_items, rustc_attrs)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub fn get_class() -> *mut () {
    unsafe extern "C" {
        #[rustc_objc_class = "MyClass"]
        safe static VAL: *mut ();
    }
    VAL
}

#[no_mangle]
pub fn get_class_again() -> *mut () {
    // Codegen should de-duplicate this class with the one from get_class above.
    unsafe extern "C" {
        #[rustc_objc_class = "MyClass"]
        safe static VAL: *mut ();
    }
    VAL
}

#[no_mangle]
pub fn get_selector() -> *mut () {
    unsafe extern "C" {
        #[rustc_objc_selector = "myMethod"]
        safe static VAL: *mut ();
    }
    VAL
}

#[no_mangle]
pub fn get_selector_again() -> *mut () {
    // Codegen should de-duplicate this selector with the one from get_selector above.
    unsafe extern "C" {
        #[rustc_objc_selector = "myMethod"]
        safe static VAL: *mut ();
    }
    VAL
}

#[no_mangle]
pub fn get_other_class() -> *mut () {
    unsafe extern "C" {
        #[rustc_objc_class = "OtherClass"]
        safe static VAL: *mut ();
    }
    VAL
}

#[no_mangle]
pub fn get_other_selector() -> *mut () {
    unsafe extern "C" {
        #[rustc_objc_selector = "otherMethod"]
        safe static VAL: *mut ();
    }
    VAL
}

// CHECK: %struct._objc_module = type { i32, i32, ptr, ptr }

// CHECK: @OBJC_CLASS_NAME_.{{[0-9]+}} = private unnamed_addr constant [8 x i8] c"MyClass\00", section "__TEXT,__cstring,cstring_literals", align 1
// CHECK: @OBJC_CLASS_REFERENCES_.{{[0-9]+}} = private global ptr @OBJC_CLASS_NAME_.{{[0-9]+}}, section "__OBJC,__cls_refs,literal_pointers,no_dead_strip", align 4
// CHECK-NOT: @OBJC_CLASS_NAME_
// CHECK-NOT: @OBJC_CLASS_REFERENCES_

// CHECK: @OBJC_METH_VAR_NAME_.{{[0-9]+}} = private unnamed_addr constant [9 x i8] c"myMethod\00", section "__TEXT,__cstring,cstring_literals", align 1
// CHECK: @OBJC_SELECTOR_REFERENCES_.{{[0-9]+}} = private externally_initialized global ptr @OBJC_METH_VAR_NAME_.{{[0-9]+}}, section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4
// CHECK-NOT: @OBJC_METH_VAR_NAME_
// CHECK-NOT: @OBJC_SELECTOR_REFERENCES_

// CHECK: @OBJC_CLASS_NAME_.{{[0-9]+}} = private unnamed_addr constant [11 x i8] c"OtherClass\00", section "__TEXT,__cstring,cstring_literals", align 1
// CHECK: @OBJC_CLASS_REFERENCES_.{{[0-9]+}} = private global ptr @OBJC_CLASS_NAME_.{{[0-9]+}}, section "__OBJC,__cls_refs,literal_pointers,no_dead_strip", align 4

// CHECK: @OBJC_METH_VAR_NAME_.{{[0-9]+}} = private unnamed_addr constant [12 x i8] c"otherMethod\00", section "__TEXT,__cstring,cstring_literals", align 1
// CHECK: @OBJC_SELECTOR_REFERENCES_.{{[0-9]+}} = private externally_initialized global ptr @OBJC_METH_VAR_NAME_.{{[0-9]+}}, section "__OBJC,__message_refs,literal_pointers,no_dead_strip", align 4

// CHECK: @OBJC_CLASS_NAME_.{{[0-9]+}} = private unnamed_addr constant [1 x i8] zeroinitializer, section "__TEXT,__cstring,cstring_literals", align 1
// CHECK: @OBJC_MODULES = private global %struct._objc_module { i32 7, i32 16, ptr @OBJC_CLASS_NAME_.{{[0-9]+}}, ptr null }, section "__OBJC,__module_info,regular,no_dead_strip", align 4

// CHECK: load ptr, ptr @OBJC_CLASS_REFERENCES_.{{[0-9]+}}, align 4
// CHECK: load ptr, ptr @OBJC_SELECTOR_REFERENCES_.{{[0-9]+}}, align 4
// CHECK: load ptr, ptr @OBJC_CLASS_REFERENCES_.{{[0-9]+}}, align 4
// CHECK: load ptr, ptr @OBJC_SELECTOR_REFERENCES_.{{[0-9]+}}, align 4

// CHECK: !{{[0-9]+}} = !{i32 1, !"Objective-C Version", i32 1}
// CHECK: !{{[0-9]+}} = !{i32 1, !"Objective-C Image Info Version", i32 0}
// CHECK: !{{[0-9]+}} = !{i32 1, !"Objective-C Image Info Section", !"__OBJC,__image_info,regular"}
// CHECK-NOT: !{{[0-9]+}} = !{i32 1, !"Objective-C Is Simulated", i32 32}
// CHECK: !{{[0-9]+}} = !{i32 1, !"Objective-C Class Properties", i32 64}
