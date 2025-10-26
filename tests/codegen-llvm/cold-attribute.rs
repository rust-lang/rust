// Checks that the cold attribute adds the llvm cold attribute.
//
//@ reference: attributes.codegen.cold.intro
//@ reference: attributes.codegen.cold.trait
//@ edition:2024
//@ compile-flags: -Copt-level=0

#![crate_type = "lib"]

// CHECK-LABEL: ; cold_attribute::free_function
// CHECK-NEXT: Function Attrs: cold {{.*}}
#[cold]
pub fn free_function() {}

// CHECK-LABEL: ; cold_attribute::async_block
// CHECK-NEXT: Function Attrs: cold {{.*}}
#[cold]
pub async fn async_block() {
    async fn x(f: impl Future<Output = ()>) {
        f.await;
    }
    x(
        // CHECK-LABEL: ; cold_attribute::async_block::{{{{closure}}}}::{{{{closure}}}}
        // CHECK-NEXT: Function Attrs: cold {{.*}}
        #[cold]
        async {},
    )
    .await;
}

pub fn closure() {
    fn x(f: impl Fn()) {
        f()
    }
    x(
        // CHECK-LABEL: ; cold_attribute::closure::{{{{closure}}}}
        // CHECK-NEXT: Function Attrs: cold {{.*}}
        #[cold]
        || {},
    );
}

pub struct S;

impl S {
    // CHECK-LABEL: ; cold_attribute::S::method
    // CHECK-NEXT: Function Attrs: cold {{.*}}
    #[cold]
    pub fn method(&self) {}
}

pub trait Trait {
    // CHECK-LABEL: ; cold_attribute::Trait::trait_fn
    // CHECK-NEXT: Function Attrs: cold {{.*}}
    #[cold]
    fn trait_fn(&self) {}

    #[cold]
    fn trait_fn_overridden(&self) {}

    fn impl_fn(&self);
}

impl Trait for S {
    // CHECK-LABEL: ; <cold_attribute::S as cold_attribute::Trait>::impl_fn
    // CHECK-NEXT: Function Attrs: cold {{.*}}
    #[cold]
    fn impl_fn(&self) {
        self.trait_fn();
    }

    // This does not have #[cold], and does not inherit the cold attribute from the trait.
    // CHECK-LABEL: ; <cold_attribute::S as cold_attribute::Trait>::trait_fn_overridden
    // CHECK: ; Function Attrs:
    // CHECK-NOT: cold
    // CHECK: define
    fn trait_fn_overridden(&self) {}
}
