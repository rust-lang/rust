// Verifies that the `#[sanitize(address = "off")]` attribute can be used to
// selectively disable sanitizer instrumentation.
//
//@ needs-sanitizer-address
//@ compile-flags: -Zsanitizer=address -Ctarget-feature=-crt-static -Copt-level=0

#![crate_type = "lib"]
#![feature(sanitize)]

// CHECK:     @UNSANITIZED = constant{{.*}} no_sanitize_address
// CHECK-NOT: @__asan_global_SANITIZED
#[no_mangle]
#[sanitize(address = "off")]
pub static UNSANITIZED: u32 = 0;

// CHECK: @__asan_global_SANITIZED
#[no_mangle]
pub static SANITIZED: u32 = 0;

// CHECK-LABEL: ; sanitize_off::unsanitized
// CHECK-NEXT:  ; Function Attrs:
// CHECK-NOT:   sanitize_address
// CHECK:       start:
// CHECK-NOT:   call void @__asan_report_load
// CHECK:       }
#[sanitize(address = "off")]
pub fn unsanitized(b: &mut u8) -> u8 {
    *b
}

// CHECK-LABEL: ; sanitize_off::sanitized
// CHECK-NEXT:  ; Function Attrs:
// CHECK:       sanitize_address
// CHECK:       start:
// CHECK:       call void @__asan_report_load
// CHECK:       }
pub fn sanitized(b: &mut u8) -> u8 {
    *b
}

#[sanitize(address = "off")]
pub mod foo {
    // CHECK-LABEL: ; sanitize_off::foo::unsanitized
    // CHECK-NEXT:  ; Function Attrs:
    // CHECK-NOT:   sanitize_address
    // CHECK:       start:
    // CHECK-NOT:   call void @__asan_report_load
    // CHECK:       }
    pub fn unsanitized(b: &mut u8) -> u8 {
        *b
    }

    // CHECK-LABEL: ; sanitize_off::foo::sanitized
    // CHECK-NEXT:  ; Function Attrs:
    // CHECK:       sanitize_address
    // CHECK:       start:
    // CHECK:       call void @__asan_report_load
    // CHECK:       }
    #[sanitize(address = "on")]
    pub fn sanitized(b: &mut u8) -> u8 {
        *b
    }
}

pub trait MyTrait {
    fn unsanitized(&self, b: &mut u8) -> u8;
    fn sanitized(&self, b: &mut u8) -> u8;

    // CHECK-LABEL: ; sanitize_off::MyTrait::unsanitized_default
    // CHECK-NEXT:  ; Function Attrs:
    // CHECK-NOT:   sanitize_address
    // CHECK:       start:
    // CHECK-NOT:   call void @__asan_report_load
    // CHECK:       }
    #[sanitize(address = "off")]
    fn unsanitized_default(&self, b: &mut u8) -> u8 {
        *b
    }

    // CHECK-LABEL: ; sanitize_off::MyTrait::sanitized_default
    // CHECK-NEXT:  ; Function Attrs:
    // CHECK:       sanitize_address
    // CHECK:       start:
    // CHECK:       call void @__asan_report_load
    // CHECK:       }
    fn sanitized_default(&self, b: &mut u8) -> u8 {
        *b
    }
}

#[sanitize(address = "off")]
impl MyTrait for () {
    // CHECK-LABEL: ; <() as sanitize_off::MyTrait>::unsanitized
    // CHECK-NEXT:  ; Function Attrs:
    // CHECK-NOT:   sanitize_address
    // CHECK:       start:
    // CHECK-NOT:   call void @__asan_report_load
    // CHECK:       }
    fn unsanitized(&self, b: &mut u8) -> u8 {
        *b
    }

    // CHECK-LABEL: ; <() as sanitize_off::MyTrait>::sanitized
    // CHECK-NEXT:  ; Function Attrs:
    // CHECK:       sanitize_address
    // CHECK:       start:
    // CHECK:       call void @__asan_report_load
    // CHECK:       }
    #[sanitize(address = "on")]
    fn sanitized(&self, b: &mut u8) -> u8 {
        *b
    }
}

pub fn expose_trait(b: &mut u8) -> u8 {
    <() as MyTrait>::unsanitized_default(&(), b);
    <() as MyTrait>::sanitized_default(&(), b)
}
