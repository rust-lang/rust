// run-pass
// aux-build:const_assert_lib.rs
use const_assert_lib::assert_same_const;

#![feature(const_option_match)]
#![feature(option_result_contains)]

assert_same_const! {
    const SOME: Option<i32> = Some(3);
    const NONE: Option<i32> = None;

    const IS_SOME_A: bool = SOME.is_some();
    const IS_SOME_B: bool = NONE.is_some();

    const IS_NONE_A: bool = SOME.is_none();
    const IS_NONE_B: bool = NONE.is_none();

    const AS_REF_A: Option<&i32> = SOME.as_ref();
    const AS_REF_B: Option<&i32> = NONE.as_ref();

    const EXPECT_A: i32 = SOME.expect("This is dangerous!");

    const UNWRAP_OR_A: i32 = SOME.unwrap_or(0);
    const UNWRAP_OR_B: i32 = NONE.unwrap_or(0);

    const OK_OR_A: Result<i32, bool> = SOME.ok_or(false);
    const OK_OR_B: Result<i32, bool> = NONE.ok_or(false);

    const AND_A: Option<bool> = SOME.and(Some(true));
    const AND_B: Option<bool> = SOME.and(None);
    const AND_C: Option<bool> = NONE.and(Some(true));
    const AND_D: Option<bool> = NONE.and(None);

    const OR_A: Option<i32> = SOME.or(Some(1));
    const OR_B: Option<i32> = SOME.or(None);
    const OR_C: Option<i32> = NONE.or(Some(1));
    const OR_D: Option<i32> = NONE.or(None);

    const XOR_A: Option<i32> = SOME.xor(Some(1));
    const XOR_B: Option<i32> = SOME.xor(None);
    const XOR_C: Option<i32> = NONE.xor(Some(1));
    const XOR_D: Option<i32> = NONE.xor(None);

    const TRANSPOSE_A: Result<Option<i32>, bool> = Some(Ok(2)).transpose();
    const TRANSPOSE_B: Result<Option<i32>, bool> = Some(Err(false)).transpose();
    const TRANSPOSE_C: Result<Option<i32>, bool> = None.transpose();
}
