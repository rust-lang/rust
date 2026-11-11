//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes -Zmir-opt-level=0

#![crate_type = "lib"]

// CHECK-LABEL: @test
#[no_mangle]
pub fn test() {
    let a = 0u8;
    &a; // keep variable in an alloca

    // CHECK: call void @llvm.lifetime.start{{.*}}({{(i[0-9 ]+, )?}}ptr %a)

    {
        let b = &Some(a);
        &b; // keep variable in an alloca

        // CHECK: call void @llvm.lifetime.start{{.*}}({{(i[0-9 ]+, )?}}{{.*}})

        // CHECK: call void @llvm.lifetime.start{{.*}}({{(i[0-9 ]+, )?}}{{.*}})

        // CHECK: call void @llvm.lifetime.end{{.*}}({{(i[0-9 ]+, )?}}{{.*}})

        // CHECK: call void @llvm.lifetime.end{{.*}}({{(i[0-9 ]+, )?}}{{.*}})
    }

    let c = 1u8;
    &c; // keep variable in an alloca

    // CHECK: call void @llvm.lifetime.start{{.*}}({{(i[0-9 ]+, )?}}ptr %c)

    // CHECK: call void @llvm.lifetime.end{{.*}}({{(i[0-9 ]+, )?}}ptr %c)

    // CHECK: call void @llvm.lifetime.end{{.*}}({{(i[0-9 ]+, )?}}ptr %a)
}
