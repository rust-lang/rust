// ignore-tidy-linelength
//@ add-core-stubs
//@ revisions: x86_64_macos
//@ [x86_64_macos] compile-flags: --target x86_64-apple-darwin
//@ [x86_64_macos] needs-llvm-components: x86
//@ revisions: aarch64_macos
//@ [aarch64_macos] compile-flags: --target aarch64-apple-darwin
//@ [aarch64_macos] needs-llvm-components: aarch64
//@ revisions: i386_ios
//@ [i386_ios] compile-flags: --target i386-apple-ios
//@ [i386_ios] needs-llvm-components: x86
//@ revisions: x86_64_ios
//@ [x86_64_ios] compile-flags: --target x86_64-apple-ios
//@ [x86_64_ios] needs-llvm-components: x86
//@ revisions: armv7s_ios
//@ [armv7s_ios] compile-flags: --target armv7s-apple-ios
//@ [armv7s_ios] needs-llvm-components: arm
//@ revisions: aarch64_ios
//@ [aarch64_ios] compile-flags: --target aarch64-apple-ios
//@ [aarch64_ios] needs-llvm-components: aarch64
//@ revisions: aarch64_ios_sim
//@ [aarch64_ios_sim] compile-flags: --target aarch64-apple-ios-sim
//@ [aarch64_ios_sim] needs-llvm-components: aarch64

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

// CHECK: %struct._class_t = type { ptr, ptr, ptr, ptr, ptr }

// CHECK: @"OBJC_CLASS_$_MyClass" = external global %struct._class_t
// CHECK: @"OBJC_CLASSLIST_REFERENCES_$_.{{[0-9]+}}" = internal global ptr @"OBJC_CLASS_$_MyClass", section "__DATA,__objc_classrefs,regular,no_dead_strip",
// x86_64_macos-SAME: align 8
// aarch64_macos-SAME: align 8
// i386_ios-SAME: align 4
// x86_64_ios-SAME: align 8
// armv7s_ios-SAME: align 4
// aarch64_ios-SAME: align 8
// aarch64_ios_sim-SAME: align 8
// CHECK-NOT: @"OBJC_CLASS_$_MyClass"
// CHECK-NOT: @"OBJC_CLASSLIST_REFERENCES_$_

// CHECK: @OBJC_METH_VAR_NAME_.{{[0-9]+}} = private unnamed_addr constant [9 x i8] c"myMethod\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK: @OBJC_SELECTOR_REFERENCES_.{{[0-9]+}} = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_.{{[0-9]+}}, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip",
// x86_64_macos-SAME: align 8
// aarch64_macos-SAME: align 8
// i386_ios-SAME: align 4
// x86_64_ios-SAME: align 8
// armv7s_ios-SAME: align 4
// aarch64_ios-SAME: align 8
// aarch64_ios_sim-SAME: align 8
// CHECK-NOT: @OBJC_METH_VAR_NAME_
// CHECK-NOT: @OBJC_SELECTOR_REFERENCES_

// CHECK: @"OBJC_CLASS_$_OtherClass" = external global %struct._class_t
// CHECK: @"OBJC_CLASSLIST_REFERENCES_$_.{{[0-9]+}}" = internal global ptr @"OBJC_CLASS_$_OtherClass", section "__DATA,__objc_classrefs,regular,no_dead_strip",
// x86_64_macos-SAME: align 8
// aarch64_macos-SAME: align 8
// i386_ios-SAME: align 4
// x86_64_ios-SAME: align 8
// armv7s_ios-SAME: align 4
// aarch64_ios-SAME: align 8
// aarch64_ios_sim-SAME: align 8

// CHECK: @OBJC_METH_VAR_NAME_.{{[0-9]+}} = private unnamed_addr constant [12 x i8] c"otherMethod\00", section "__TEXT,__objc_methname,cstring_literals", align 1
// CHECK: @OBJC_SELECTOR_REFERENCES_.{{[0-9]+}} = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_.{{[0-9]+}}, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip",
// x86_64_macos-SAME: align 8
// aarch64_macos-SAME: align 8
// i386_ios-SAME: align 4
// x86_64_ios-SAME: align 8
// armv7s_ios-SAME: align 4
// aarch64_ios-SAME: align 8
// aarch64_ios_sim-SAME: align 8

// CHECK-NOT: @OBJC_CLASS_NAME_
// CHECK-NOT: @OBJC_MODULES

// CHECK: load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.{{[0-9]+}}",
// x86_64_macos-SAME: align 8
// aarch64_macos-SAME: align 8
// i386_ios-SAME: align 4
// x86_64_ios-SAME: align 8
// armv7s_ios-SAME: align 4
// aarch64_ios-SAME: align 8
// aarch64_ios_sim-SAME: align 8

// CHECK: load ptr, ptr @OBJC_SELECTOR_REFERENCES_.{{[0-9]+}},
// x86_64_macos-SAME: align 8
// aarch64_macos-SAME: align 8
// i386_ios-SAME: align 4
// x86_64_ios-SAME: align 8
// armv7s_ios-SAME: align 4
// aarch64_ios-SAME: align 8
// aarch64_ios_sim-SAME: align 8

// CHECK: load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.{{[0-9]+}}",
// x86_64_macos-SAME: align 8
// aarch64_macos-SAME: align 8
// i386_ios-SAME: align 4
// x86_64_ios-SAME: align 8
// armv7s_ios-SAME: align 4
// aarch64_ios-SAME: align 8
// aarch64_ios_sim-SAME: align 8

// CHECK: load ptr, ptr @OBJC_SELECTOR_REFERENCES_.{{[0-9]+}},
// x86_64_macos-SAME: align 8
// aarch64_macos-SAME: align 8
// i386_ios-SAME: align 4
// x86_64_ios-SAME: align 8
// armv7s_ios-SAME: align 4
// aarch64_ios-SAME: align 8
// aarch64_ios_sim-SAME: align 8

// CHECK: !{{[0-9]+}} = !{i32 1, !"Objective-C Version", i32 2}
// CHECK: !{{[0-9]+}} = !{i32 1, !"Objective-C Image Info Version", i32 0}
// CHECK: !{{[0-9]+}} = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}

// x86_64_macos-NOT: !{{[0-9]+}} = !{i32 1, !"Objective-C Is Simulated", i32 32}
// aarch64_macos-NOT: !{{[0-9]+}} = !{i32 1, !"Objective-C Is Simulated", i32 32}
// i386_ios: !{{[0-9]+}} = !{i32 1, !"Objective-C Is Simulated", i32 32}
// x86_64_ios: !{{[0-9]+}} = !{i32 1, !"Objective-C Is Simulated", i32 32}
// armv7s_ios-NOT: !{{[0-9]+}} = !{i32 1, !"Objective-C Is Simulated", i32 32}
// aarch64_ios-NOT: !{{[0-9]+}} = !{i32 1, !"Objective-C Is Simulated", i32 32}
// aarch64_ios_sim: !{{[0-9]+}} = !{i32 1, !"Objective-C Is Simulated", i32 32}

// CHECK: !{{[0-9]+}} = !{i32 1, !"Objective-C Class Properties", i32 64}
