// run-pass
#![feature(const_option_match)]
#![feature(option_result_contains)]

macro_rules! assert_same_const {
    ($(const $ident:ident: $ty:ty = $exp:expr;)+) => {
        $(const $ident: $ty = $exp;)+

        pub fn main() {
            $(assert_eq!($exp, $ident);)+
        }
    }
}

// These functions let us use the functional interfaces of Option (like unwrap_or_else, map_or,
// map, etc.) without using closures, which aren't implemented in const contexts yet; see
// https://github.com/rust-lang/rust/issues/63997

const fn is_zero(i: i32) -> bool {
    i == 0i32
}

const fn get_zero() -> i32 {
    0
}

const fn get_false() -> bool {
    false
}

const fn get_some() -> Option<i32> {
    Some(2)
}

const fn get_none() -> Option<i32> {
    None
}

const fn is_pos(i: &i32) -> bool {
    i.is_positive()
}

const fn is_neg(i: &i32) -> bool {
    i.is_negative()
}

const fn i32_to_some(i: i32) -> Option<i32> {
    Some(i * 2)
}

const fn i32_to_none(_i: i32) -> Option<i32> {
    None
}

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

    const UNWRAP_OR_ELSE_A: i32 = SOME.unwrap_or_else(get_zero);
    const UNWRAP_OR_ELSE_B: i32 = NONE.unwrap_or_else(get_zero);

    const MAP_A: Option<bool> = SOME.map(is_zero);
    const MAP_B: Option<bool> = NONE.map(is_zero);

    const MAP_OR_A: bool = SOME.map_or(false, is_zero);
    const MAP_OR_B: bool = NONE.map_or(false, is_zero);

    const MAP_OR_ELSE_A: bool = SOME.map_or_else(get_false, is_zero);
    const MAP_OR_ELSE_B: bool = NONE.map_or_else(get_false, is_zero);

    const OK_OR_A: Result<i32, bool> = SOME.ok_or(false);
    const OK_OR_B: Result<i32, bool> = NONE.ok_or(false);

    const OK_OR_ELSE_A: Result<i32, bool> = SOME.ok_or_else(get_false);
    const OK_OR_ELSE_B: Result<i32, bool> = NONE.ok_or_else(get_false);

    const AND_A: Option<bool> = SOME.and(Some(true));
    const AND_B: Option<bool> = SOME.and(None);
    const AND_C: Option<bool> = NONE.and(Some(true));
    const AND_D: Option<bool> = NONE.and(None);

    const AND_THEN_A: Option<i32> = SOME.and_then(i32_to_some);
    const AND_THEN_B: Option<i32> = SOME.and_then(i32_to_none);
    const AND_THEN_C: Option<i32> = NONE.and_then(i32_to_some);
    const AND_THEN_D: Option<i32> = NONE.and_then(i32_to_none);

    const FILTER_A: Option<i32> = SOME.filter(is_pos);
    const FILTER_B: Option<i32> = SOME.filter(is_neg);
    const FILTER_C: Option<i32> = NONE.filter(is_pos);
    const FILTER_D: Option<i32> = NONE.filter(is_neg);

    const OR_A: Option<i32> = SOME.or(Some(1));
    const OR_B: Option<i32> = SOME.or(None);
    const OR_C: Option<i32> = NONE.or(Some(1));
    const OR_D: Option<i32> = NONE.or(None);

    const OR_ELSE_A: Option<i32> = SOME.or_else(get_some);
    const OR_ELSE_B: Option<i32> = SOME.or_else(get_none);
    const OR_ELSE_C: Option<i32> = NONE.or_else(get_some);
    const OR_ELSE_D: Option<i32> = NONE.or_else(get_none);

    const XOR_A: Option<i32> = SOME.xor(Some(1));
    const XOR_B: Option<i32> = SOME.xor(None);
    const XOR_C: Option<i32> = NONE.xor(Some(1));
    const XOR_D: Option<i32> = NONE.xor(None);
}
