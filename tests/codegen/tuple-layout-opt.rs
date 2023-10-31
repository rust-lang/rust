// ignore-emscripten
// compile-flags: -C no-prepopulate-passes -Copt-level=0

// Test that tuples get optimized layout, in particular with a ZST in the last field (#63244)

#![crate_type="lib"]

type ScalarZstLast = (u128, ());
// CHECK: define {{(dso_local )?}}i128 @test_ScalarZstLast(i128 %_1)
#[no_mangle]
pub fn test_ScalarZstLast(_: ScalarZstLast) -> ScalarZstLast { loop {} }

type ScalarZstFirst = ((), u128);
// CHECK: define {{(dso_local )?}}i128 @test_ScalarZstFirst(i128 %_1)
#[no_mangle]
pub fn test_ScalarZstFirst(_: ScalarZstFirst) -> ScalarZstFirst { loop {} }

type ScalarPairZstLast = (u8, u128, ());
// CHECK: define {{(dso_local )?}}{ i128, i8 } @test_ScalarPairZstLast(i128 %_1.0, i8 %_1.1)
#[no_mangle]
pub fn test_ScalarPairZstLast(_: ScalarPairZstLast) -> ScalarPairZstLast { loop {} }

type ScalarPairZstFirst = ((), u8, u128);
// CHECK: define {{(dso_local )?}}{ i8, i128 } @test_ScalarPairZstFirst(i8 %_1.0, i128 %_1.1)
#[no_mangle]
pub fn test_ScalarPairZstFirst(_: ScalarPairZstFirst) -> ScalarPairZstFirst { loop {} }

type ScalarPairLotsOfZsts = ((), u8, (), u128, ());
// CHECK: define {{(dso_local )?}}{ i128, i8 } @test_ScalarPairLotsOfZsts(i128 %_1.0, i8 %_1.1)
#[no_mangle]
pub fn test_ScalarPairLotsOfZsts(_: ScalarPairLotsOfZsts) -> ScalarPairLotsOfZsts { loop {} }

type ScalarPairLottaNesting = (((), ((), u8, (), u128, ())), ());
// CHECK: define {{(dso_local )?}}{ i128, i8 } @test_ScalarPairLottaNesting(i128 %_1.0, i8 %_1.1)
#[no_mangle]
pub fn test_ScalarPairLottaNesting(_: ScalarPairLottaNesting) -> ScalarPairLottaNesting { loop {} }
