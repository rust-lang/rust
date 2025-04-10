//@ compile-flags: -Copt-level=1
//@ only-64bit

#![crate_type = "lib"]
#![feature(core_intrinsics)]

// Check each of the 3 cases for `codegen_get_discr`.

// FIXME: once our min-bar LLVM has `range` attributes, update the various
// tests here to no longer have the `range`s and `nsw`s as optional.

// Case 0: One tagged variant.
pub enum Enum0 {
    A(bool),
    B,
}

// CHECK-LABEL: define noundef{{( range\(i8 [0-9]+, [0-9]+\))?}} i8 @match0(i8{{.+}}%0)
// CHECK-NEXT: start:
// CHECK-NEXT: %[[IS_B:.+]] = icmp eq i8 %0, 2
// CHECK-NEXT: %[[TRUNC:.+]] = and i8 %0, 1
// CHECK-NEXT: %[[R:.+]] = select i1 %[[IS_B]], i8 13, i8 %[[TRUNC]]
// CHECK-NEXT: ret i8 %[[R]]
#[no_mangle]
pub fn match0(e: Enum0) -> u8 {
    use Enum0::*;
    match e {
        A(b) => b as u8,
        B => 13,
    }
}

// Case 1: Niche values are on a boundary for `range`.
pub enum Enum1 {
    A(bool),
    B,
    C,
}

// CHECK-LABEL: define noundef{{( range\(i8 [0-9]+, [0-9]+\))?}} i8 @match1(i8{{.+}}%0)
// CHECK-NEXT: start:
// CHECK-NEXT: %[[REL_VAR:.+]] = add{{( nsw)?}} i8 %0, -2
// CHECK-NEXT: %[[REL_VAR_WIDE:.+]] = zext i8 %[[REL_VAR]] to i64
// CHECK-NEXT: %[[IS_NICHE:.+]] = icmp ult i8 %[[REL_VAR]], 2
// CHECK-NEXT: %[[NICHE_DISCR:.+]] = add nuw nsw i64 %[[REL_VAR_WIDE]], 1
// CHECK-NEXT: %[[DISCR:.+]] = select i1 %[[IS_NICHE]], i64 %[[NICHE_DISCR]], i64 0
// CHECK-NEXT: switch i64 %[[DISCR]]
#[no_mangle]
pub fn match1(e: Enum1) -> u8 {
    use Enum1::*;
    match e {
        A(b) => b as u8,
        B => 13,
        C => 100,
    }
}

// Case 2: Special cases don't apply.
#[rustfmt::skip]
pub enum X {
    _2=2, _3, _4, _5, _6, _7, _8, _9, _10, _11,
    _12, _13, _14, _15, _16, _17, _18, _19, _20,
    _21, _22, _23, _24, _25, _26, _27, _28, _29,
    _30, _31, _32, _33, _34, _35, _36, _37, _38,
    _39, _40, _41, _42, _43, _44, _45, _46, _47,
    _48, _49, _50, _51, _52, _53, _54, _55, _56,
    _57, _58, _59, _60, _61, _62, _63, _64, _65,
    _66, _67, _68, _69, _70, _71, _72, _73, _74,
    _75, _76, _77, _78, _79, _80, _81, _82, _83,
    _84, _85, _86, _87, _88, _89, _90, _91, _92,
    _93, _94, _95, _96, _97, _98, _99, _100, _101,
    _102, _103, _104, _105, _106, _107, _108, _109,
    _110, _111, _112, _113, _114, _115, _116, _117,
    _118, _119, _120, _121, _122, _123, _124, _125,
    _126, _127, _128, _129, _130, _131, _132, _133,
    _134, _135, _136, _137, _138, _139, _140, _141,
    _142, _143, _144, _145, _146, _147, _148, _149,
    _150, _151, _152, _153, _154, _155, _156, _157,
    _158, _159, _160, _161, _162, _163, _164, _165,
    _166, _167, _168, _169, _170, _171, _172, _173,
    _174, _175, _176, _177, _178, _179, _180, _181,
    _182, _183, _184, _185, _186, _187, _188, _189,
    _190, _191, _192, _193, _194, _195, _196, _197,
    _198, _199, _200, _201, _202, _203, _204, _205,
    _206, _207, _208, _209, _210, _211, _212, _213,
    _214, _215, _216, _217, _218, _219, _220, _221,
    _222, _223, _224, _225, _226, _227, _228, _229,
    _230, _231, _232, _233, _234, _235, _236, _237,
    _238, _239, _240, _241, _242, _243, _244, _245,
    _246, _247, _248, _249, _250, _251, _252, _253,
}

pub enum Enum2 {
    A(X),
    B,
    C,
    D,
    E,
}

// CHECK-LABEL: define noundef{{( range\(i8 [0-9]+, [0-9]+\))?}} i8 @match2(i8{{.+}}%0)
// CHECK-NEXT: start:
// CHECK-NEXT: %[[REL_VAR:.+]] = add i8 %0, 2
// CHECK-NEXT: %[[REL_VAR_WIDE:.+]] = zext i8 %[[REL_VAR]] to i64
// CHECK-NEXT: %[[IS_NICHE:.+]] = icmp ult i8 %[[REL_VAR]], 4
// CHECK-NEXT: %[[NICHE_DISCR:.+]] = add nuw nsw i64 %[[REL_VAR_WIDE]], 1
// CHECK-NEXT: %[[DISCR:.+]] = select i1 %[[IS_NICHE]], i64 %[[NICHE_DISCR]], i64 0
// CHECK-NEXT: switch i64 %[[DISCR]]
#[no_mangle]
pub fn match2(e: Enum2) -> u8 {
    use Enum2::*;
    match e {
        A(b) => b as u8,
        B => 13,
        C => 100,
        D => 200,
        E => 250,
    }
}

// And make sure it works even if the niched scalar is a pointer.
// (For example, that we don't try to `sub` on pointers.)

// CHECK-LABEL: define noundef{{( range\(i16 -?[0-9]+, -?[0-9]+\))?}} i16 @match3(ptr{{.+}}%0)
// CHECK-NEXT: start:
// CHECK-NEXT: %[[IS_NULL:.+]] = icmp eq ptr %0, null
// CHECK-NEXT: br i1 %[[IS_NULL]]
#[no_mangle]
pub fn match3(e: Option<&u8>) -> i16 {
    match e {
        Some(r) => *r as _,
        None => -1,
    }
}

// If the untagged variant is in the middle, there's an impossible value that's
// not reflected in the `range` parameter attribute, so we assume it away.

#[derive(PartialEq)]
pub enum MiddleNiche {
    A,
    B,
    C(bool),
    D,
    E,
}

// CHECK-LABEL: define noundef{{( range\(i8 -?[0-9]+, -?[0-9]+\))?}} i8 @match4(i8{{.+}}%0)
// CHECK-NEXT: start:
// CHECK-NEXT: %[[REL_VAR:.+]] = add{{( nsw)?}} i8 %0, -2
// CHECK-NEXT: %[[IS_NICHE:.+]] = icmp ult i8 %[[REL_VAR]], 5
// CHECK-NEXT: %[[NOT_IMPOSSIBLE:.+]] = icmp ne i8 %[[REL_VAR]], 2
// CHECK-NEXT: call void @llvm.assume(i1 %[[NOT_IMPOSSIBLE]])
// CHECK-NEXT: %[[DISCR:.+]] = select i1 %[[IS_NICHE]], i8 %[[REL_VAR]], i8 2
// CHECK-NEXT: switch i8 %[[DISCR]]
#[no_mangle]
pub fn match4(e: MiddleNiche) -> u8 {
    use MiddleNiche::*;
    match e {
        A => 13,
        B => 100,
        C(b) => b as u8,
        D => 200,
        E => 250,
    }
}

// CHECK-LABEL: define{{.+}}i1 @match4_is_c(i8{{.+}}%e)
// CHECK-NEXT: start
// CHECK-NEXT: %[[REL_VAR:.+]] = add{{( nsw)?}} i8 %e, -2
// CHECK-NEXT: %[[NOT_NICHE:.+]] = icmp ugt i8 %[[REL_VAR]], 4
// CHECK-NEXT: %[[NOT_IMPOSSIBLE:.+]] = icmp ne i8 %[[REL_VAR]], 2
// CHECK-NEXT: call void @llvm.assume(i1 %[[NOT_IMPOSSIBLE]])
// CHECK-NEXT: ret i1 %[[NOT_NICHE]]
#[no_mangle]
pub fn match4_is_c(e: MiddleNiche) -> bool {
    // Before #139098, this couldn't optimize out the `select` because it looked
    // like it was possible for a `2` to be produced on both sides.

    std::intrinsics::discriminant_value(&e) == 2
}

// You have to do something pretty obnoxious to get a variant index that doesn't
// fit in the tag size, but it's possible

pub enum Never {}

pub enum HugeVariantIndex {
    V000(Never),
    V001(Never),
    V002(Never),
    V003(Never),
    V004(Never),
    V005(Never),
    V006(Never),
    V007(Never),
    V008(Never),
    V009(Never),
    V010(Never),
    V011(Never),
    V012(Never),
    V013(Never),
    V014(Never),
    V015(Never),
    V016(Never),
    V017(Never),
    V018(Never),
    V019(Never),
    V020(Never),
    V021(Never),
    V022(Never),
    V023(Never),
    V024(Never),
    V025(Never),
    V026(Never),
    V027(Never),
    V028(Never),
    V029(Never),
    V030(Never),
    V031(Never),
    V032(Never),
    V033(Never),
    V034(Never),
    V035(Never),
    V036(Never),
    V037(Never),
    V038(Never),
    V039(Never),
    V040(Never),
    V041(Never),
    V042(Never),
    V043(Never),
    V044(Never),
    V045(Never),
    V046(Never),
    V047(Never),
    V048(Never),
    V049(Never),
    V050(Never),
    V051(Never),
    V052(Never),
    V053(Never),
    V054(Never),
    V055(Never),
    V056(Never),
    V057(Never),
    V058(Never),
    V059(Never),
    V060(Never),
    V061(Never),
    V062(Never),
    V063(Never),
    V064(Never),
    V065(Never),
    V066(Never),
    V067(Never),
    V068(Never),
    V069(Never),
    V070(Never),
    V071(Never),
    V072(Never),
    V073(Never),
    V074(Never),
    V075(Never),
    V076(Never),
    V077(Never),
    V078(Never),
    V079(Never),
    V080(Never),
    V081(Never),
    V082(Never),
    V083(Never),
    V084(Never),
    V085(Never),
    V086(Never),
    V087(Never),
    V088(Never),
    V089(Never),
    V090(Never),
    V091(Never),
    V092(Never),
    V093(Never),
    V094(Never),
    V095(Never),
    V096(Never),
    V097(Never),
    V098(Never),
    V099(Never),
    V100(Never),
    V101(Never),
    V102(Never),
    V103(Never),
    V104(Never),
    V105(Never),
    V106(Never),
    V107(Never),
    V108(Never),
    V109(Never),
    V110(Never),
    V111(Never),
    V112(Never),
    V113(Never),
    V114(Never),
    V115(Never),
    V116(Never),
    V117(Never),
    V118(Never),
    V119(Never),
    V120(Never),
    V121(Never),
    V122(Never),
    V123(Never),
    V124(Never),
    V125(Never),
    V126(Never),
    V127(Never),
    V128(Never),
    V129(Never),
    V130(Never),
    V131(Never),
    V132(Never),
    V133(Never),
    V134(Never),
    V135(Never),
    V136(Never),
    V137(Never),
    V138(Never),
    V139(Never),
    V140(Never),
    V141(Never),
    V142(Never),
    V143(Never),
    V144(Never),
    V145(Never),
    V146(Never),
    V147(Never),
    V148(Never),
    V149(Never),
    V150(Never),
    V151(Never),
    V152(Never),
    V153(Never),
    V154(Never),
    V155(Never),
    V156(Never),
    V157(Never),
    V158(Never),
    V159(Never),
    V160(Never),
    V161(Never),
    V162(Never),
    V163(Never),
    V164(Never),
    V165(Never),
    V166(Never),
    V167(Never),
    V168(Never),
    V169(Never),
    V170(Never),
    V171(Never),
    V172(Never),
    V173(Never),
    V174(Never),
    V175(Never),
    V176(Never),
    V177(Never),
    V178(Never),
    V179(Never),
    V180(Never),
    V181(Never),
    V182(Never),
    V183(Never),
    V184(Never),
    V185(Never),
    V186(Never),
    V187(Never),
    V188(Never),
    V189(Never),
    V190(Never),
    V191(Never),
    V192(Never),
    V193(Never),
    V194(Never),
    V195(Never),
    V196(Never),
    V197(Never),
    V198(Never),
    V199(Never),
    V200(Never),
    V201(Never),
    V202(Never),
    V203(Never),
    V204(Never),
    V205(Never),
    V206(Never),
    V207(Never),
    V208(Never),
    V209(Never),
    V210(Never),
    V211(Never),
    V212(Never),
    V213(Never),
    V214(Never),
    V215(Never),
    V216(Never),
    V217(Never),
    V218(Never),
    V219(Never),
    V220(Never),
    V221(Never),
    V222(Never),
    V223(Never),
    V224(Never),
    V225(Never),
    V226(Never),
    V227(Never),
    V228(Never),
    V229(Never),
    V230(Never),
    V231(Never),
    V232(Never),
    V233(Never),
    V234(Never),
    V235(Never),
    V236(Never),
    V237(Never),
    V238(Never),
    V239(Never),
    V240(Never),
    V241(Never),
    V242(Never),
    V243(Never),
    V244(Never),
    V245(Never),
    V246(Never),
    V247(Never),
    V248(Never),
    V249(Never),
    V250(Never),
    V251(Never),
    V252(Never),
    V253(Never),
    V254(Never),
    V255(Never),
    V256(Never),

    Possible257,
    Bool258(bool),
    Possible259,
}

// CHECK-LABEL: define noundef{{( range\(i8 [0-9]+, [0-9]+\))?}} i8 @match5(i8{{.+}}%0)
// CHECK-NEXT: start:
// CHECK-NEXT: %[[REL_VAR:.+]] = add{{( nsw)?}} i8 %0, -2
// CHECK-NEXT: %[[REL_VAR_WIDE:.+]] = zext i8 %[[REL_VAR]] to i64
// CHECK-NEXT: %[[IS_NICHE:.+]] = icmp ult i8 %[[REL_VAR]], 3
// CHECK-NEXT: %[[NOT_IMPOSSIBLE:.+]] = icmp ne i8 %[[REL_VAR]], 1
// CHECK-NEXT: call void @llvm.assume(i1 %[[NOT_IMPOSSIBLE]])
// CHECK-NEXT: %[[NICHE_DISCR:.+]] = add nuw nsw i64 %[[REL_VAR_WIDE]], 257
// CHECK-NEXT: %[[DISCR:.+]] = select i1 %[[IS_NICHE]], i64 %[[NICHE_DISCR]], i64 258
// CHECK-NEXT: switch i64 %[[DISCR]],
// CHECK-NEXT:   i64 257,
// CHECK-NEXT:   i64 258,
// CHECK-NEXT:   i64 259,
#[no_mangle]
pub fn match5(e: HugeVariantIndex) -> u8 {
    use HugeVariantIndex::*;
    match e {
        Possible257 => 13,
        Bool258(b) => b as u8,
        Possible259 => 100,
    }
}
