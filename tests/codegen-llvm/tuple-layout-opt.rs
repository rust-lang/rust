// 32-bit systems will return 128bit values using a return area pointer.
//@ revisions: bit32 bit64
//@[bit32] only-32bit
//@[bit64] only-64bit
//@ compile-flags: -C no-prepopulate-passes -Copt-level=0

// Test that tuples get optimized layout, in particular with a ZST in the last field (#63244)

#![crate_type = "lib"]

type ScalarZstLast = (u128, ());
// bit32: define {{(dso_local )?}}void @test_ScalarZstLast({{.*}} sret([16 x i8]) {{.*}}, i128 %_1)
// bit64: define {{(dso_local )?}}i128 @test_ScalarZstLast(i128 %_1)
#[no_mangle]
pub fn test_ScalarZstLast(_: ScalarZstLast) -> ScalarZstLast {
    loop {}
}

type ScalarZstFirst = ((), u128);
// bit32: define {{(dso_local )?}}void @test_ScalarZstFirst({{.*}} sret([16 x i8]) {{.*}}, i128 %_1)
// bit64: define {{(dso_local )?}}i128 @test_ScalarZstFirst(i128 %_1)
#[no_mangle]
pub fn test_ScalarZstFirst(_: ScalarZstFirst) -> ScalarZstFirst {
    loop {}
}

type ScalarPairZstLast = (u8, u128, ());
// CHECK: define {{(dso_local )?}}void @test_ScalarPairZstLast(ptr sret({{[^,]*}})
// CHECK-SAME: %_0, i128 %_1.0, i8 %_1.1)
#[no_mangle]
pub fn test_ScalarPairZstLast(_: ScalarPairZstLast) -> ScalarPairZstLast {
    loop {}
}

type ScalarPairZstFirst = ((), u8, u128);
// CHECK: define {{(dso_local )?}}void @test_ScalarPairZstFirst(ptr sret({{[^,]*}})
// CHECK-SAME: %_0, i8 %_1.0, i128 %_1.1)
#[no_mangle]
pub fn test_ScalarPairZstFirst(_: ScalarPairZstFirst) -> ScalarPairZstFirst {
    loop {}
}

type ScalarPairLotsOfZsts = ((), u8, (), u128, ());
// CHECK: define {{(dso_local )?}}void @test_ScalarPairLotsOfZsts(ptr sret({{[^,]*}})
// CHECK-SAME: %_0, i128 %_1.0, i8 %_1.1)
#[no_mangle]
pub fn test_ScalarPairLotsOfZsts(_: ScalarPairLotsOfZsts) -> ScalarPairLotsOfZsts {
    loop {}
}

type ScalarPairLottaNesting = (((), ((), u8, (), u128, ())), ());
// CHECK: define {{(dso_local )?}}void @test_ScalarPairLottaNesting(ptr sret({{[^,]*}})
// CHECK-SAME: %_0, i128 %_1.0, i8 %_1.1)
#[no_mangle]
pub fn test_ScalarPairLottaNesting(_: ScalarPairLottaNesting) -> ScalarPairLottaNesting {
    loop {}
}
