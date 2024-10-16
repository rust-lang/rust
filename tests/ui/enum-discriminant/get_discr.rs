//@ run-pass
use std::mem;

// Now that there are several variations on the code generated in
// `codegen_get_discr`, let's make sure the various cases yield the correct
// result.

// To get the discriminant of an E<X1> value, there are no shortcuts - we must
// do the full algorithm.
// `X1` is u8 with two niche values.
#[derive(Clone, Copy)]
#[repr(u8)]
pub enum X1 {
    _1 = 1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16,
    _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32,
    _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48,
    _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64,
    _65, _66, _67, _68, _69, _70, _71, _72, _73, _74, _75, _76, _77, _78, _79, _80,
    _81, _82, _83, _84, _85, _86, _87, _88, _89, _90, _91, _92, _93, _94, _95, _96,
    _97, _98, _99, _100, _101, _102, _103, _104, _105, _106, _107, _108, _109, _110, _111, _112,
    _113, _114, _115, _116, _117, _118, _119, _120, _121, _122, _123, _124, _125, _126, _127, _128,
    _129, _130, _131, _132, _133, _134, _135, _136, _137, _138, _139, _140, _141, _142, _143, _144,
    _145, _146, _147, _148, _149, _150, _151, _152, _153, _154, _155, _156, _157, _158, _159, _160,
    _161, _162, _163, _164, _165, _166, _167, _168, _169, _170, _171, _172, _173, _174, _175, _176,
    _177, _178, _179, _180, _181, _182, _183, _184, _185, _186, _187, _188, _189, _190, _191, _192,
    _193, _194, _195, _196, _197, _198, _199, _200, _201, _202, _203, _204, _205, _206, _207, _208,
    _209, _210, _211, _212, _213, _214, _215, _216, _217, _218, _219, _220, _221, _222, _223, _224,
    _225, _226, _227, _228, _229, _230, _231, _232, _233, _234, _235, _236, _237, _238, _239, _240,
    _241, _242, _243, _244, _245, _246, _247, _248, _249, _250, _251, _252, _253, _254,
}

#[repr(i8)]
pub enum X2 {
    _1 = -1, _2 = 0, _3 = 1,
}

#[derive(Clone, Copy)]
#[repr(i8)]
pub enum X3 {
    _1 = -128, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16,
    _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32,
    _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48,
    _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64,
    _65, _66, _67, _68, _69, _70, _71, _72, _73, _74, _75, _76, _77, _78, _79, _80,
    _81, _82, _83, _84, _85, _86, _87, _88, _89, _90, _91, _92, _93, _94, _95, _96,
    _97, _98, _99, _100, _101, _102, _103, _104, _105, _106, _107, _108, _109, _110, _111, _112,
    _113, _114, _115, _116, _117, _118, _119, _120, _121, _122, _123, _124, _125, _126, _127, _128,
    _129, _130, _131, _132, _133, _134, _135, _136, _137, _138, _139, _140, _141, _142, _143, _144,
    _145, _146, _147, _148, _149, _150, _151, _152, _153, _154, _155, _156, _157, _158, _159, _160,
    _161, _162, _163, _164, _165, _166, _167, _168, _169, _170, _171, _172, _173, _174, _175, _176,
    _177, _178, _179, _180, _181, _182, _183, _184, _185, _186, _187, _188, _189, _190, _191, _192,
    _193, _194, _195, _196, _197, _198, _199, _200, _201, _202, _203, _204, _205, _206, _207, _208,
    _209, _210, _211, _212, _213, _214, _215, _216, _217, _218, _219, _220, _221, _222, _223, _224,
    _225, _226, _227, _228, _229, _230, _231, _232, _233, _234, _235, _236, _237, _238, _239, _240,
    _241, _242, _243, _244, _245, _246, _247, _248, _249, _250, _251, _252, _253, _254,
}

#[repr(i8)]
pub enum X4 {
    _1 = -126, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16,
    _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32,
    _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48,
    _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64,
    _65, _66, _67, _68, _69, _70, _71, _72, _73, _74, _75, _76, _77, _78, _79, _80,
    _81, _82, _83, _84, _85, _86, _87, _88, _89, _90, _91, _92, _93, _94, _95, _96,
    _97, _98, _99, _100, _101, _102, _103, _104, _105, _106, _107, _108, _109, _110, _111, _112,
    _113, _114, _115, _116, _117, _118, _119, _120, _121, _122, _123, _124, _125, _126, _127, _128,
    _129, _130, _131, _132, _133, _134, _135, _136, _137, _138, _139, _140, _141, _142, _143, _144,
    _145, _146, _147, _148, _149, _150, _151, _152, _153, _154, _155, _156, _157, _158, _159, _160,
    _161, _162, _163, _164, _165, _166, _167, _168, _169, _170, _171, _172, _173, _174, _175, _176,
    _177, _178, _179, _180, _181, _182, _183, _184, _185, _186, _187, _188, _189, _190, _191, _192,
    _193, _194, _195, _196, _197, _198, _199, _200, _201, _202, _203, _204, _205, _206, _207, _208,
    _209, _210, _211, _212, _213, _214, _215, _216, _217, _218, _219, _220, _221, _222, _223, _224,
    _225, _226, _227, _228, _229, _230, _231, _232, _233, _234, _235, _236, _237, _238, _239, _240,
    _241, _242, _243, _244, _245, _246, _247, _248, _249, _250, _251, _252, _253, _254,
}

pub enum E<X> {
    A(X),
    B,
    C,
}

pub fn match_e<X>(e: E<X>) -> u8 {
    use E::*;
    match e {
        A(_) => 0,
        B => 1,
        C => 2,
    }
}

#[derive(Clone, Copy)]
pub enum Void {}
// Special case that there's only one tagged variant.
#[derive(Clone, Copy)]
pub enum E1<X> {
    A(X),   // the untagged variant.
    B,
}

pub const fn match_e1<X: Copy>(e: E1<X>) -> u8 {
    use E1::*;
    match e {
        A(_) => 0,
        B => 1,
    }
}

#[derive(Clone, Copy)]
pub enum E1WithAbsent<X> {
    V1(Void),
    A(X),
    V2(Void),
    B,
    V3(Void),
}

pub const fn match_e1_with_absent<X: Copy>(e: E1WithAbsent<X>) -> u8 {
    use E1WithAbsent::*;
    match e {
        A(_) => 0,
        B => 1,
        _ => unreachable!(),
    }
}

// General case. And the untagged variant is contained in the niche_variants.
#[derive(Clone, Copy)]
pub enum E2<X> {
    A,
    B(X),
    C,
}

pub const fn match_e2<X: Copy>(e: E2<X>) -> u8 {
    use E2::*;
    match e {
        A => 0,
        B(_) => 1,
        C => 2,
    }
}

#[derive(Clone, Copy)]
pub enum E2WithAbsent<X> {
    V1(Void),
    V2(Void),
    A,
    B(X),
    C,
    V3(Void),
}

pub const fn match_e2_with_absent<X: Copy>(e: E2WithAbsent<X>) -> u8 {
    use E2WithAbsent::*;
    match e {
        A => 0,
        B(_) => 1,
        C => 2,
        _ => unreachable!(),
    }
}

// General case. And the untagged variant is not contained in the niche_variants.
#[derive(Clone, Copy)]
pub enum E3<X> {
    A,
    B,
    C(X)
}

pub const fn match_e3<X: Copy>(e: E3<X>) -> u8 {
    use E3::*;
    match e {
        A => 0,
        B => 1,
        C(_) => 2,
    }
}

#[derive(Clone, Copy)]
pub enum E3WithAbsent<X> {
    V1(Void),
    V2(Void),
    A,
    B,
    C(X),
    V3(Void),
}

pub const fn match_e3_with_absent<X: Copy>(e: E3WithAbsent<X>) -> u8 {
    use E3WithAbsent::*;
    match e {
        A => 0,
        B => 1,
        C(_) => 2,
        _ => unreachable!(),
    }
}


fn main() {
    assert_eq!(match_e(E::A(X1::_1)), 0);
    assert_eq!(match_e(E::A(X1::_2)), 0);
    assert_eq!(match_e(E::A(X1::_254)), 0);
    assert_eq!(match_e(E::<X1>::B), 1);
    assert_eq!(match_e(E::<X1>::C), 2);
    assert_eq!(match_e(E::A(X2::_1)), 0);
    assert_eq!(match_e(E::A(X2::_2)), 0);
    assert_eq!(match_e(E::A(X2::_3)), 0);
    assert_eq!(match_e(E::<X2>::B), 1);
    assert_eq!(match_e(E::<X2>::C), 2);
    assert_eq!(match_e(E::A(X3::_1)), 0);
    assert_eq!(match_e(E::A(X3::_2)), 0);
    assert_eq!(match_e(E::A(X3::_254)), 0);
    assert_eq!(match_e(E::<X3>::B), 1);
    assert_eq!(match_e(E::<X3>::C), 2);
    assert_eq!(match_e(E::A(X4::_1)), 0);
    assert_eq!(match_e(E::A(X4::_2)), 0);
    assert_eq!(match_e(E::A(X4::_254)), 0);
    assert_eq!(match_e(E::<X4>::B), 1);
    assert_eq!(match_e(E::<X4>::C), 2);
    assert_eq!(match_e(E::A(false)), 0);
    assert_eq!(match_e(E::A(true)), 0);
    assert_eq!(match_e(E::<bool>::B), 1);
    assert_eq!(match_e(E::<bool>::C), 2);

    // Check `u8` primitive type as discriminant.
    assert_eq!(mem::size_of::<E1<X1>>(), 1);
    assert_eq!(match_e1(E1::A(X1::_1)), 0);
    assert_eq!(match_e1(E1::A(X1::_2)), 0);
    assert_eq!(match_e1(E1::A(X1::_254)), 0);
    assert_eq!(match_e1(E1::<X1>::B), 1);
    assert_eq!(mem::size_of::<E1WithAbsent<X1>>(), 1);
    assert_eq!(match_e1_with_absent(E1WithAbsent::A(X1::_1)), 0);
    assert_eq!(match_e1_with_absent(E1WithAbsent::A(X1::_2)), 0);
    assert_eq!(match_e1_with_absent(E1WithAbsent::A(X1::_254)), 0);
    assert_eq!(match_e1_with_absent(E1WithAbsent::<X1>::B), 1);

    // Check `i8` primitive type as discriminant.
    assert_eq!(mem::size_of::<E1<X3>>(), 1);
    assert_eq!(match_e1(E1::A(X3::_1)), 0);
    assert_eq!(match_e1(E1::A(X3::_2)), 0);
    assert_eq!(match_e1(E1::A(X3::_254)), 0);
    assert_eq!(match_e1(E1::<X3>::B), 1);
    assert_eq!(mem::size_of::<E1WithAbsent<X3>>(), 1);
    assert_eq!(match_e1_with_absent(E1WithAbsent::A(X3::_1)), 0);
    assert_eq!(match_e1_with_absent(E1WithAbsent::A(X3::_2)), 0);
    assert_eq!(match_e1_with_absent(E1WithAbsent::A(X3::_254)), 0);
    assert_eq!(match_e1_with_absent(E1WithAbsent::<X3>::B), 1);

    assert_eq!(mem::size_of::<E2<X1>>(), 1);
    assert_eq!(match_e2(E2::<X1>::A), 0);
    assert_eq!(match_e2(E2::B(X1::_1)), 1);
    assert_eq!(match_e2(E2::B(X1::_2)), 1);
    assert_eq!(match_e2(E2::B(X1::_254)), 1);
    assert_eq!(match_e2(E2::<X1>::C), 2);
    assert_eq!(mem::size_of::<E2WithAbsent<X1>>(), 1);
    assert_eq!(match_e2_with_absent(E2WithAbsent::<X1>::A), 0);
    assert_eq!(match_e2_with_absent(E2WithAbsent::B(X1::_1)), 1);
    assert_eq!(match_e2_with_absent(E2WithAbsent::B(X1::_2)), 1);
    assert_eq!(match_e2_with_absent(E2WithAbsent::B(X1::_254)), 1);
    assert_eq!(match_e2_with_absent(E2WithAbsent::<X1>::C), 2);

    assert_eq!(mem::size_of::<E2<X3>>(), 1);
    assert_eq!(match_e2(E2::<X3>::A), 0);
    assert_eq!(match_e2(E2::B(X3::_1)), 1);
    assert_eq!(match_e2(E2::B(X3::_2)), 1);
    assert_eq!(match_e2(E2::B(X3::_254)), 1);
    assert_eq!(match_e2(E2::<X3>::C), 2);
    assert_eq!(mem::size_of::<E2WithAbsent<X3>>(), 1);
    assert_eq!(match_e2_with_absent(E2WithAbsent::<X3>::A), 0);
    assert_eq!(match_e2_with_absent(E2WithAbsent::B(X3::_1)), 1);
    assert_eq!(match_e2_with_absent(E2WithAbsent::B(X3::_2)), 1);
    assert_eq!(match_e2_with_absent(E2WithAbsent::B(X3::_254)), 1);
    assert_eq!(match_e2_with_absent(E2WithAbsent::<X3>::C), 2);

    assert_eq!(mem::size_of::<E3<X1>>(), 1);
    assert_eq!(match_e3(E3::<X1>::A), 0);
    assert_eq!(match_e3(E3::<X1>::B), 1);
    assert_eq!(match_e3(E3::C(X1::_1)), 2);
    assert_eq!(match_e3(E3::C(X1::_2)), 2);
    assert_eq!(match_e3(E3::C(X1::_254)), 2);
    assert_eq!(mem::size_of::<E3WithAbsent<X1>>(), 1);
    assert_eq!(match_e3_with_absent(E3WithAbsent::<X1>::A), 0);
    assert_eq!(match_e3_with_absent(E3WithAbsent::<X1>::B), 1);
    assert_eq!(match_e3_with_absent(E3WithAbsent::C(X1::_1)), 2);
    assert_eq!(match_e3_with_absent(E3WithAbsent::C(X1::_2)), 2);
    assert_eq!(match_e3_with_absent(E3WithAbsent::C(X1::_254)), 2);

    assert_eq!(mem::size_of::<E3<X3>>(), 1);
    assert_eq!(match_e3(E3::<X3>::A), 0);
    assert_eq!(match_e3(E3::<X3>::B), 1);
    assert_eq!(match_e3(E3::C(X3::_1)), 2);
    assert_eq!(match_e3(E3::C(X3::_2)), 2);
    assert_eq!(match_e3(E3::C(X3::_254)), 2);
    assert_eq!(mem::size_of::<E3WithAbsent<X3>>(), 1);
    assert_eq!(match_e3_with_absent(E3WithAbsent::<X3>::A), 0);
    assert_eq!(match_e3_with_absent(E3WithAbsent::<X3>::B), 1);
    assert_eq!(match_e3_with_absent(E3WithAbsent::C(X3::_1)), 2);
    assert_eq!(match_e3_with_absent(E3WithAbsent::C(X3::_2)), 2);
    assert_eq!(match_e3_with_absent(E3WithAbsent::C(X3::_254)), 2);

    // Check set_discr and get_discr work as intended in const eval too.
    const _: () = {
        // Check `u8` primitive type as discriminant.
        assert!(mem::size_of::<E1<X1>>() == 1);
        assert!(match_e1(E1::A(X1::_1)) == 0);
        assert!(match_e1(E1::A(X1::_2)) == 0);
        assert!(match_e1(E1::A(X1::_254)) == 0);
        assert!(match_e1(E1::<X1>::B) == 1);
        assert!(mem::size_of::<E1WithAbsent<X1>>() == 1);
        assert!(match_e1_with_absent(E1WithAbsent::A(X1::_1)) == 0);
        assert!(match_e1_with_absent(E1WithAbsent::A(X1::_2)) == 0);
        assert!(match_e1_with_absent(E1WithAbsent::A(X1::_254)) == 0);
        assert!(match_e1_with_absent(E1WithAbsent::<X1>::B) == 1);

        // Check `i8` primitive type as discriminant.
        assert!(mem::size_of::<E1<X3>>() == 1);
        assert!(match_e1(E1::A(X3::_1)) == 0);
        assert!(match_e1(E1::A(X3::_2)) == 0);
        assert!(match_e1(E1::A(X3::_254)) == 0);
        assert!(match_e1(E1::<X3>::B) == 1);
        assert!(mem::size_of::<E1WithAbsent<X3>>() == 1);
        assert!(match_e1_with_absent(E1WithAbsent::A(X3::_1)) == 0);
        assert!(match_e1_with_absent(E1WithAbsent::A(X3::_2)) == 0);
        assert!(match_e1_with_absent(E1WithAbsent::A(X3::_254)) == 0);
        assert!(match_e1_with_absent(E1WithAbsent::<X3>::B) == 1);

        assert!(mem::size_of::<E2<X1>>() == 1);
        assert!(match_e2(E2::<X1>::A) == 0);
        assert!(match_e2(E2::B(X1::_1)) == 1);
        assert!(match_e2(E2::B(X1::_2)) == 1);
        assert!(match_e2(E2::B(X1::_254)) == 1);
        assert!(match_e2(E2::<X1>::C) == 2);
        assert!(mem::size_of::<E2WithAbsent<X1>>() == 1);
        assert!(match_e2_with_absent(E2WithAbsent::<X1>::A) == 0);
        assert!(match_e2_with_absent(E2WithAbsent::B(X1::_1)) == 1);
        assert!(match_e2_with_absent(E2WithAbsent::B(X1::_2)) == 1);
        assert!(match_e2_with_absent(E2WithAbsent::B(X1::_254)) == 1);
        assert!(match_e2_with_absent(E2WithAbsent::<X1>::C) == 2);

        assert!(mem::size_of::<E2<X3>>() == 1);
        assert!(match_e2(E2::<X3>::A) == 0);
        assert!(match_e2(E2::B(X3::_1)) == 1);
        assert!(match_e2(E2::B(X3::_2)) == 1);
        assert!(match_e2(E2::B(X3::_254)) == 1);
        assert!(match_e2(E2::<X3>::C) == 2);
        assert!(mem::size_of::<E2WithAbsent<X3>>() == 1);
        assert!(match_e2_with_absent(E2WithAbsent::<X3>::A) == 0);
        assert!(match_e2_with_absent(E2WithAbsent::B(X3::_1)) == 1);
        assert!(match_e2_with_absent(E2WithAbsent::B(X3::_2)) == 1);
        assert!(match_e2_with_absent(E2WithAbsent::B(X3::_254)) == 1);
        assert!(match_e2_with_absent(E2WithAbsent::<X3>::C) == 2);

        assert!(mem::size_of::<E3<X1>>() == 1);
        assert!(match_e3(E3::<X1>::A) == 0);
        assert!(match_e3(E3::<X1>::B) == 1);
        assert!(match_e3(E3::C(X1::_1)) == 2);
        assert!(match_e3(E3::C(X1::_2)) == 2);
        assert!(match_e3(E3::C(X1::_254)) == 2);
        assert!(mem::size_of::<E3WithAbsent<X1>>() == 1);
        assert!(match_e3_with_absent(E3WithAbsent::<X1>::A) == 0);
        assert!(match_e3_with_absent(E3WithAbsent::<X1>::B) == 1);
        assert!(match_e3_with_absent(E3WithAbsent::C(X1::_1)) == 2);
        assert!(match_e3_with_absent(E3WithAbsent::C(X1::_2)) == 2);
        assert!(match_e3_with_absent(E3WithAbsent::C(X1::_254)) == 2);

        assert!(mem::size_of::<E3<X3>>() == 1);
        assert!(match_e3(E3::<X3>::A) == 0);
        assert!(match_e3(E3::<X3>::B) == 1);
        assert!(match_e3(E3::C(X3::_1)) == 2);
        assert!(match_e3(E3::C(X3::_2)) == 2);
        assert!(match_e3(E3::C(X3::_254)) == 2);
        assert!(mem::size_of::<E3WithAbsent<X3>>() == 1);
        assert!(match_e3_with_absent(E3WithAbsent::<X3>::A) == 0);
        assert!(match_e3_with_absent(E3WithAbsent::<X3>::B) == 1);
        assert!(match_e3_with_absent(E3WithAbsent::C(X3::_1)) == 2);
        assert!(match_e3_with_absent(E3WithAbsent::C(X3::_2)) == 2);
        assert!(match_e3_with_absent(E3WithAbsent::C(X3::_254)) == 2);
    };
}
