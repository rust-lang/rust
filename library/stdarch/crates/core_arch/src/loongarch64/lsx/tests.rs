// This code is automatically generated. DO NOT MODIFY.
// See crates/stdarch-gen-loongarch/README.md

use crate::{
    core_arch::{loongarch64::*, simd::*},
    mem::transmute,
};
use stdarch_test::simd_test;

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsll_b() {
    let a = i8x16::new(
        -96, 33, -12, -39, 82, 20, 52, 0, -99, -60, -50, -85, -6, -83, -52, -23,
    );
    let b = i8x16::new(
        50, 37, 88, 105, -45, -52, 119, 2, 19, 109, 95, 116, -101, -126, -104, -119,
    );
    let r = i64x2::new(70990221811840, -3257029622096690968);

    assert_eq!(r, transmute(lsx_vsll_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsll_h() {
    let a = i16x8::new(2551, -25501, -5868, -8995, 27363, 18426, -10212, -26148);
    let b = i16x8::new(-10317, -20778, -9962, -8975, 25298, 12929, -13803, -18669);
    let r = i64x2::new(-5063658964307128392, -3539825456407336052);

    assert_eq!(r, transmute(lsx_vsll_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsll_w() {
    let a = i32x4::new(1371197240, -1100536513, 781269067, -294302078);
    let b = i32x4::new(82237029, -819106294, -96895338, -456101700);
    let r = i64x2::new(-7163824029380778240, 2305843009528266752);

    assert_eq!(r, transmute(lsx_vsll_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsll_d() {
    let a = i64x2::new(5700293115058898640, 9057986892130087440);
    let b = i64x2::new(8592669249977019309, -1379694176202045825);
    let r = i64x2::new(1790743801833193472, 0);

    assert_eq!(r, transmute(lsx_vsll_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslli_b() {
    let a = i8x16::new(
        90, 123, 29, -67, 120, -106, 104, -39, -62, -56, -92, -75, 113, 123, -120, -52,
    );
    let r = i64x2::new(-2780807324588213414, -3708578564830607166);

    assert_eq!(r, transmute(lsx_vslli_b::<0>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslli_h() {
    let a = i16x8::new(18469, -14840, 23655, -3474, 7467, 2798, -15418, 26847);
    let r = i64x2::new(-7241759886206301888, 4017476402818337472);

    assert_eq!(r, transmute(lsx_vslli_h::<6>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslli_w() {
    let a = i32x4::new(20701902, -1777432355, 6349179, 1747667894);
    let r = i64x2::new(4189319625752393728, -5967594959501136896);

    assert_eq!(r, transmute(lsx_vslli_w::<10>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslli_d() {
    let a = i64x2::new(-5896889635782282086, -8807609320972692839);
    let r = i64x2::new(-4233027607937510592, -5142337165482896608);

    assert_eq!(r, transmute(lsx_vslli_d::<5>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsra_b() {
    let a = i8x16::new(
        0, 72, -102, -88, 101, -100, 66, -113, 68, -13, 2, 4, -61, 66, -24, 72,
    );
    let b = i8x16::new(
        34, 5, 102, 83, -87, 43, 94, 107, -84, 88, -103, 5, 127, 43, -28, -69,
    );
    let r = i64x2::new(-1080315035391229440, 720022881735668484);

    assert_eq!(r, transmute(lsx_vsra_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsra_h() {
    let a = i16x8::new(29313, 15702, 30839, 9343, -19597, 5316, -32305, -13755);
    let b = i16x8::new(14017, 3796, 23987, -27244, -13363, 21333, -10262, 23633);
    let r = i64x2::new(164116464290576704, -1935703552267190275);

    assert_eq!(r, transmute(lsx_vsra_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsra_w() {
    let a = i32x4::new(-309802992, -833530117, -1757716660, 1577882592);
    let b = i32x4::new(-670772992, 2044335288, -1224858031, 520588790);
    let r = i64x2::new(-210763200496, 1619202657181);

    assert_eq!(r, transmute(lsx_vsra_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsra_d() {
    let a = i64x2::new(-1372092312892164486, 6937900992858870877);
    let b = i64x2::new(4251079558060308329, 4657697142994416829);
    let r = i64x2::new(-623956, 3);

    assert_eq!(r, transmute(lsx_vsra_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrai_b() {
    let a = i8x16::new(
        -4, 92, -7, -110, 81, -20, -18, -113, 43, 110, -105, 53, -101, -100, -56, -120,
    );
    let r = i64x2::new(-2018743940785760257, -2093355901512246518);

    assert_eq!(r, transmute(lsx_vsrai_b::<2>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrai_h() {
    let a = i16x8::new(-22502, -7299, 19084, -21578, -28082, 20851, 23456, 15524);
    let r = i64x2::new(-1688828385492998, 844446405361657);

    assert_eq!(r, transmute(lsx_vsrai_h::<12>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrai_w() {
    let a = i32x4::new(743537539, 1831641900, -1639033567, -984629971);
    let r = i64x2::new(30008936499988, -16131897170029);

    assert_eq!(r, transmute(lsx_vsrai_w::<18>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrai_d() {
    let a = i64x2::new(-8375997486414293750, 1714581574012370587);
    let r = i64x2::new(-476121, 97462);

    assert_eq!(r, transmute(lsx_vsrai_d::<44>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrar_b() {
    let a = i8x16::new(
        123, 17, -3, 27, 49, 89, -61, 105, -77, 87, 87, 15, -113, 75, -69, 40,
    );
    let b = i8x16::new(
        14, 5, 123, -33, 72, -126, -70, -33, -124, -55, -82, -78, -33, -12, -25, -114,
    );
    let r = i64x2::new(139917463134404866, 143840305941130491);

    assert_eq!(r, transmute(lsx_vsrar_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrar_h() {
    let a = i16x8::new(-25154, -18230, -10510, -29541, 25913, 29143, 21372, 14979);
    let b = i16x8::new(-26450, 2176, 31587, 2222, 13726, 30172, 1067, -14273);
    let r = i64x2::new(-287115463426050, 42950131714);

    assert_eq!(r, transmute(lsx_vsrar_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrar_w() {
    let a = i32x4::new(-139995520, 1671693163, -640570871, 2138298219);
    let b = i32x4::new(-1532076758, 940127488, 1781366421, 1497262222);
    let r = i64x2::new(7179867468326627830, 560544771735247);

    assert_eq!(r, transmute(lsx_vsrar_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrar_d() {
    let a = i64x2::new(-489385672013329488, -1253364580216579403);
    let b = i64x2::new(3571440266112779495, -725943254065719378);
    let r = i64x2::new(-890187, -17811);

    assert_eq!(r, transmute(lsx_vsrar_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrari_b() {
    let a = i8x16::new(
        -20, 33, -49, -120, -30, -40, 67, 93, -77, -2, 16, -36, 108, -107, 23, -53,
    );
    let r = i64x2::new(867219992078845182, -503291487652282122);

    assert_eq!(r, transmute(lsx_vsrari_b::<3>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrari_h() {
    let a = i16x8::new(29939, -1699, 12357, 30805, -30883, 31936, 15701, -11818);
    let r = i64x2::new(4222154715365391, -1688815499411471);

    assert_eq!(r, transmute(lsx_vsrari_h::<11>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrari_w() {
    let a = i32x4::new(588196178, -1058764534, 1325397591, 1169671026);
    let r = i64x2::new(-4294967295, 4294967297);

    assert_eq!(r, transmute(lsx_vsrari_w::<30>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrari_d() {
    let a = i64x2::new(-2795326946470057100, 6746045132217841338);
    let r = i64x2::new(-174707934154378569, 421627820763615084);

    assert_eq!(r, transmute(lsx_vsrari_d::<4>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrl_b() {
    let a = i8x16::new(
        73, 74, 66, -104, -30, 25, 93, -107, 105, -89, -115, -22, -94, -36, -55, -28,
    );
    let b = i8x16::new(
        81, 13, -9, -46, -24, 0, 91, 123, 90, -52, -24, 56, 64, -4, -66, -17,
    );
    let r = i64x2::new(1300161376517358116, 72917012339034650);

    assert_eq!(r, transmute(lsx_vsrl_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrl_h() {
    let a = i16x8::new(29049, 13489, 20776, -12268, 25704, -28758, -6146, -27463);
    let b = i16x8::new(16605, -13577, -26644, -17739, 11000, -29283, -15971, 20169);
    let r = i64x2::new(468374382728249347, 20829178341621860);

    assert_eq!(r, transmute(lsx_vsrl_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrl_w() {
    let a = i32x4::new(-2108561731, -402290458, -1418385618, 1489749824);
    let b = i32x4::new(1777885221, -1725401090, 1849724045, -1051851102);
    let r = i64x2::new(12953227061, 1599606693325790121);

    assert_eq!(r, transmute(lsx_vsrl_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrl_d() {
    let a = i64x2::new(2854528248771186187, 804951867404831945);
    let b = i64x2::new(-7903128394835365398, 7601347629202818185);
    let r = i64x2::new(649044, 1572171616025062);

    assert_eq!(r, transmute(lsx_vsrl_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrli_b() {
    let a = i8x16::new(
        84, -108, 98, 45, 126, -124, 105, 108, 0, 61, -29, -31, -75, -41, 114, -33,
    );
    let r = i64x2::new(1952909805632365845, 3971107439766933248);

    assert_eq!(r, transmute(lsx_vsrli_b::<2>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrli_h() {
    let a = i16x8::new(29545, 354, 27695, 20915, -32766, -24491, 10641, 20310);
    let r = i64x2::new(11259230996660281, 10977609996304448);

    assert_eq!(r, transmute(lsx_vsrli_h::<9>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrli_w() {
    let a = i32x4::new(627703601, 922874410, -234412645, -1216101872);
    let r = i64x2::new(3870813506329215, 12913695352717769);

    assert_eq!(r, transmute(lsx_vsrli_w::<10>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrli_d() {
    let a = i64x2::new(1407685950714554203, -6076144426076800688);
    let r = i64x2::new(9, 85);

    assert_eq!(r, transmute(lsx_vsrli_d::<57>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrlr_b() {
    let a = i8x16::new(
        -79, 91, -123, 112, -84, 70, -78, -74, -104, 27, -94, -46, -49, -78, 113, -2,
    );
    let b = i8x16::new(
        23, 4, -120, -11, -13, 103, 84, 58, -108, 121, -66, -9, -81, 91, 71, -33,
    );
    let r = i64x2::new(3317746744565237249, 144420860932066826);

    assert_eq!(r, transmute(lsx_vsrlr_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrlr_h() {
    let a = i16x8::new(14153, -26873, 3115, 28304, 4881, -8446, 28628, 8837);
    let b = i16x8::new(19500, -26403, -1282, 12290, -18989, 25105, -24347, 6707);
    let r = i64x2::new(1991716935204929539, 311033695131730530);

    assert_eq!(r, transmute(lsx_vsrlr_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrlr_w() {
    let a = i32x4::new(1997879294, 120007491, -1807289594, -1854395615);
    let b = i32x4::new(1830015593, -1452673200, 962662328, -252736055);
    let r = i64x2::new(7864089021084, 20473000998469780);

    assert_eq!(r, transmute(lsx_vsrlr_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrlr_d() {
    let a = i64x2::new(5993546441420611680, 4358546479290416194);
    let b = i64x2::new(-1543621369665313706, 8544381131364512650);
    let r = i64x2::new(1428972826343, 4256393046182047);

    assert_eq!(r, transmute(lsx_vsrlr_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrlri_b() {
    let a = i8x16::new(
        -41, 87, -43, -35, 79, -10, -103, 1, 52, -35, 8, -17, -116, 84, -91, 51,
    );
    let r = i64x2::new(93866580842851436, 1896906350202744602);

    assert_eq!(r, transmute(lsx_vsrlri_b::<1>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrlri_h() {
    let a = i16x8::new(-18045, 1968, 22966, 3692, 2010, -17108, 3373, -30706);
    let r = i64x2::new(1039304252363684227, -8642956144778934310);

    assert_eq!(r, transmute(lsx_vsrlri_h::<0>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrlri_w() {
    let a = i32x4::new(1306456564, -1401620667, -839707416, -1634862919);
    let r = i64x2::new(1553353645217275455, 1428132662790218397);

    assert_eq!(r, transmute(lsx_vsrlri_w::<3>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrlri_d() {
    let a = i64x2::new(-3683179565838693027, 6160461828074490983);
    let r = i64x2::new(205, 85);

    assert_eq!(r, transmute(lsx_vsrlri_d::<56>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitclr_b() {
    let a = u8x16::new(
        238, 18, 41, 55, 84, 12, 87, 155, 124, 76, 175, 240, 181, 121, 58, 183,
    );
    let b = u8x16::new(
        57, 132, 149, 173, 76, 177, 99, 144, 8, 167, 2, 144, 70, 60, 105, 232,
    );
    let r = i64x2::new(-7325372782311046420, -5316383129963115396);

    assert_eq!(r, transmute(lsx_vbitclr_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitclr_h() {
    let a = u16x8::new(14340, 59474, 49868, 46012, 53117, 6307, 22589, 53749);
    let b = u16x8::new(26587, 57597, 34751, 38678, 23919, 45729, 62569, 5978);
    let r = i64x2::new(-5495443997997256700, -3317648531059028099);

    assert_eq!(r, transmute(lsx_vbitclr_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitclr_w() {
    let a = u32x4::new(1581022148, 2519245321, 296293885, 127383934);
    let b = u32x4::new(1968231094, 2827365864, 4097273355, 4016923215);
    let r = i64x2::new(-7626667807832507452, 546969093373761021);

    assert_eq!(r, transmute(lsx_vbitclr_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitclr_d() {
    let a = u64x2::new(17203892527896963423, 12937109545250696056);
    let b = u64x2::new(5723204188033770667, 2981956604140378920);
    let r = i64x2::new(-1242851545812588193, -5509634528458855560);

    assert_eq!(r, transmute(lsx_vbitclr_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitclri_b() {
    let a = u8x16::new(
        146, 23, 223, 183, 109, 56, 35, 105, 178, 156, 170, 57, 196, 164, 185, 161,
    );
    let r = i64x2::new(7503621968728299154, -6865556469255070542);

    assert_eq!(r, transmute(lsx_vbitclri_b::<0>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitclri_h() {
    let a = u16x8::new(17366, 58985, 22108, 45942, 27326, 19605, 9632, 32322);
    let r = i64x2::new(-5515130134779575338, 8809640793386347198);

    assert_eq!(r, transmute(lsx_vbitclri_h::<10>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitclri_w() {
    let a = u32x4::new(718858183, 3771164920, 1842485081, 896350597);
    let r = i64x2::new(-2249714073768237625, 3849796501707560281);

    assert_eq!(r, transmute(lsx_vbitclri_w::<9>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitclri_d() {
    let a = u64x2::new(10838658690401820648, 3833745076866321369);
    let r = i64x2::new(-7608085933063544856, 3833744527110507481);

    assert_eq!(r, transmute(lsx_vbitclri_d::<39>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitset_b() {
    let a = u8x16::new(
        229, 230, 162, 180, 94, 215, 193, 145, 28, 90, 35, 171, 225, 7, 84, 128,
    );
    let b = u8x16::new(
        209, 178, 73, 112, 118, 233, 139, 239, 2, 23, 209, 152, 236, 51, 195, 75,
    );
    let r = i64x2::new(-7941579666116909337, -8620998056061183460);

    assert_eq!(r, transmute(lsx_vbitset_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitset_h() {
    let a = u16x8::new(967, 49899, 53264, 29198, 56634, 42461, 51022, 31627);
    let b = u16x8::new(64512, 23847, 57770, 47705, 8024, 31966, 14493, 50266);
    let r = i64x2::new(8218739538452480967, 9190693790629616954);

    assert_eq!(r, transmute(lsx_vbitset_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitset_w() {
    let a = u32x4::new(2899706360, 1274114722, 1170526770, 3308854969);
    let b = u32x4::new(3259082048, 1303228302, 1429001720, 209615081);
    let r = i64x2::new(5472281065241838073, -4235320193476931022);

    assert_eq!(r, transmute(lsx_vbitset_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitset_d() {
    let a = u64x2::new(8117422063017946604, 5026948610774344635);
    let b = u64x2::new(12687331714071910183, 1753585392879336372);
    let r = i64x2::new(8117422612773760492, 5031452210401715131);

    assert_eq!(r, transmute(lsx_vbitset_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitseti_b() {
    let a = u8x16::new(
        163, 123, 56, 129, 159, 111, 214, 85, 141, 240, 190, 190, 175, 215, 20, 81,
    );
    let r = i64x2::new(6185254145054243811, 5860546440891134157);

    assert_eq!(r, transmute(lsx_vbitseti_b::<6>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitseti_h() {
    let a = u16x8::new(15222, 59961, 52253, 2908, 61562, 41309, 63627, 4191);
    let r = i64x2::new(819316619673811830, 1179934905985921146);

    assert_eq!(r, transmute(lsx_vbitseti_h::<1>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitseti_w() {
    let a = u32x4::new(3788412756, 1863556832, 1913138259, 1199998627);
    let r = i64x2::new(8012922850722617172, 5162962059379878995);

    assert_eq!(r, transmute(lsx_vbitseti_w::<21>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitseti_d() {
    let a = u64x2::new(10744510173660993785, 16946223211744108759);
    let r = i64x2::new(-7702233900048557831, -1500520861831225129);

    assert_eq!(r, transmute(lsx_vbitseti_d::<27>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitrev_b() {
    let a = u8x16::new(
        50, 114, 173, 149, 9, 38, 147, 232, 52, 235, 56, 98, 113, 120, 249, 238,
    );
    let b = u8x16::new(
        252, 187, 218, 48, 148, 63, 222, 247, 56, 181, 124, 130, 243, 202, 86, 253,
    );
    let r = i64x2::new(7553563628828981794, -3550669970358088907);

    assert_eq!(r, transmute(lsx_vbitrev_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitrev_h() {
    let a = u16x8::new(8304, 965, 30335, 58555, 41304, 8461, 30573, 59417);
    let b = u16x8::new(21347, 23131, 57157, 13786, 34463, 33445, 23964, 48087);
    let r = i64x2::new(-2253077037977362312, -1686202867067838120);

    assert_eq!(r, transmute(lsx_vbitrev_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitrev_w() {
    let a = u32x4::new(3821500454, 1067219398, 1766391845, 676798616);
    let b = u32x4::new(3330530584, 4153020036, 822570638, 2652744506);
    let r = i64x2::new(4583672484591007782, 3195058299616182309);

    assert_eq!(r, transmute(lsx_vbitrev_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitrev_d() {
    let a = u64x2::new(16016664040604304047, 18062107512190600767);
    let b = u64x2::new(10942298949673565895, 12884740754463765660);
    let r = i64x2::new(-2430080033105247697, -384636561250515393);

    assert_eq!(r, transmute(lsx_vbitrev_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitrevi_b() {
    let a = u8x16::new(
        184, 147, 93, 34, 212, 175, 25, 125, 50, 34, 160, 241, 228, 231, 77, 110,
    );
    let r = i64x2::new(8727320563398842300, 7658903196653594166);

    assert_eq!(r, transmute(lsx_vbitrevi_b::<2>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitrevi_h() {
    let a = u16x8::new(15083, 24599, 61212, 12408, 48399, 59833, 45416, 58826);
    let r = i64x2::new(8104420064785562347, -6500117680329458417);

    assert_eq!(r, transmute(lsx_vbitrevi_h::<14>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitrevi_w() {
    let a = u32x4::new(1200613355, 1418062686, 3847355950, 3312937419);
    let r = i64x2::new(6099540060505368555, -4226793400815190482);

    assert_eq!(r, transmute(lsx_vbitrevi_w::<21>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitrevi_d() {
    let a = u64x2::new(295858379748270823, 1326723086853575042);
    let r = i64x2::new(295858379748254439, 1326723086853591426);

    assert_eq!(r, transmute(lsx_vbitrevi_d::<14>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vadd_b() {
    let a = i8x16::new(
        14, -124, 73, 125, 119, 60, 127, -10, 31, 89, 50, -88, 29, -28, -53, -8,
    );
    let b = i8x16::new(
        94, -52, -56, 75, -104, 77, 16, 82, 82, 69, -81, -75, 25, -102, -109, 23,
    );
    let r = i64x2::new(5228548393274527852, 1107461330348121713);

    assert_eq!(r, transmute(lsx_vadd_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vadd_h() {
    let a = i16x8::new(14051, -27363, -25412, -27329, 25098, 5182, -13698, -15422);
    let b = i16x8::new(-25040, 15453, -28080, -31322, -24429, -12453, -18073, 27019);
    let r = i64x2::new(1938006946753467667, 3264410328302682781);

    assert_eq!(r, transmute(lsx_vadd_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vadd_w() {
    let a = i32x4::new(-724548235, -1051318497, -203352059, 1502361914);
    let b = i32x4::new(-1169804484, 389773725, -731843701, -1825112934);
    let r = i64x2::new(-2841313158179161935, -1386205072290870384);

    assert_eq!(r, transmute(lsx_vadd_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vadd_d() {
    let a = i64x2::new(-7298628992874088690, 8943248591432696479);
    let b = i64x2::new(7093939531558864473, 4047047970310912233);
    let r = i64x2::new(-204689461315224217, -5456447511965942904);

    assert_eq!(r, transmute(lsx_vadd_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddi_bu() {
    let a = i8x16::new(
        -126, 4, -123, -78, -37, -26, -41, -119, -16, -82, 33, 59, -110, -98, 26, -6,
    );
    let r = i64x2::new(-7790681010872578420, 298548864442153210);

    assert_eq!(r, transmute(lsx_vaddi_bu::<10>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddi_hu() {
    let a = i16x8::new(-16986, -28417, 11657, 16608, -30167, 18602, 8897, -854);
    let r = i64x2::new(4681541984598867390, -233585914045887935);

    assert_eq!(r, transmute(lsx_vaddi_hu::<24>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddi_wu() {
    let a = i32x4::new(1142343549, 56714754, -180143297, 408668191);
    let r = i64x2::new(243588023362963327, 1755216527965240129);

    assert_eq!(r, transmute(lsx_vaddi_wu::<2>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddi_du() {
    let a = i64x2::new(4516502893749962130, 9158051921593642947);
    let r = i64x2::new(4516502893749962139, 9158051921593642956);

    assert_eq!(r, transmute(lsx_vaddi_du::<9>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsub_b() {
    let a = i8x16::new(
        125, 95, 56, 31, 69, -81, 65, -123, -72, 14, -43, 81, -12, -107, 106, 3,
    );
    let b = i8x16::new(
        -80, 10, -21, 84, -99, 8, 125, -66, 79, -71, 123, 61, 61, -31, 41, -118,
    );
    let r = i64x2::new(-4051929421319416371, 8737463450488952169);

    assert_eq!(r, transmute(lsx_vsub_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsub_h() {
    let a = i16x8::new(-17949, -2606, 1774, 18199, 28344, 28423, 16206, 25414);
    let b = i16x8::new(15368, 16207, 9677, 21447, -29583, -22036, 1845, 15671);
    let r = i64x2::new(-913983189443969573, 2742472381424198215);

    assert_eq!(r, transmute(lsx_vsub_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsub_w() {
    let a = i32x4::new(678216285, 1230738403, -1278396773, -1257816042);
    let b = i32x4::new(617176389, -1376778690, 1463940361, 620446698);
    let r = i64x2::new(-7247543435452521192, -8067077040042720878);

    assert_eq!(r, transmute(lsx_vsub_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsub_d() {
    let a = i64x2::new(7239192343295591267, -5127457864580422409);
    let b = i64x2::new(1314101702815749241, 7673634401554993450);
    let r = i64x2::new(5925090640479842026, 5645651807574135757);

    assert_eq!(r, transmute(lsx_vsub_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsubi_bu() {
    let a = i8x16::new(
        -83, 36, 83, -2, 40, -92, 98, -95, -24, 113, 46, -20, 120, -93, 28, 85,
    );
    let r = i64x2::new(-8192169673836457574, 4758493248402185941);

    assert_eq!(r, transmute(lsx_vsubi_bu::<19>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsubi_hu() {
    let a = i16x8::new(13272, -26858, -235, 16054, 29698, 1377, 4604, -3878);
    let r = i64x2::new(4514576075959186376, -1096043853912116238);

    assert_eq!(r, transmute(lsx_vsubi_hu::<16>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsubi_wu() {
    let a = i32x4::new(1277091145, -2076591216, -1523555105, -945754023);
    let r = i64x2::new(-8918891362898748088, -4061982600368986914);

    assert_eq!(r, transmute(lsx_vsubi_wu::<1>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsubi_du() {
    let a = i64x2::new(-8248876128472283209, -2119651236628000925);
    let r = i64x2::new(-8248876128472283234, -2119651236628000950);

    assert_eq!(r, transmute(lsx_vsubi_du::<25>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmax_b() {
    let a = i8x16::new(
        -120, -51, 13, 82, 100, 7, 127, 17, -89, -95, -45, 121, 64, -60, 89, 105,
    );
    let b = i8x16::new(
        -47, -64, 96, 41, -30, -122, 3, -7, 123, -96, 68, 36, 14, 31, 74, -22,
    );
    let r = i64x2::new(1260734548147228113, 7591133008682590587);

    assert_eq!(r, transmute(lsx_vmax_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmax_h() {
    let a = i16x8::new(-14821, -29280, 26700, -12293, 2186, -23309, 13454, -1630);
    let b = i16x8::new(25637, -11569, -23103, 6983, -17125, 5183, -709, 5986);
    let r = i64x2::new(1965654441534120997, 1684966995419662474);

    assert_eq!(r, transmute(lsx_vmax_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmax_w() {
    let a = i32x4::new(-2113940850, -647459228, -686153447, 852904547);
    let b = i32x4::new(643859790, -389733899, -1309288060, 1934346522);
    let r = i64x2::new(-1673894349703707314, 8307955054730158361);

    assert_eq!(r, transmute(lsx_vmax_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmax_d() {
    let a = i64x2::new(-990960773872867733, 6406870358170165030);
    let b = i64x2::new(-6137495199657896371, 2160025776787809810);
    let r = i64x2::new(-990960773872867733, 6406870358170165030);

    assert_eq!(r, transmute(lsx_vmax_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaxi_b() {
    let a = i8x16::new(
        -67, 109, 33, -22, -96, 84, -56, 81, 122, 23, -70, -71, -42, 108, -50, 23,
    );
    let r = i64x2::new(5908253215318699518, 1728939149412407162);

    assert_eq!(r, transmute(lsx_vmaxi_b::<-2>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaxi_h() {
    let a = i16x8::new(-14059, 19536, 15816, 28251, 23079, -10486, -11781, 25565);
    let r = i64x2::new(7952017497535807498, 7195907822558272039);

    assert_eq!(r, transmute(lsx_vmaxi_h::<10>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaxi_w() {
    let a = i32x4::new(-1136628686, -168033999, -2082324641, -1789957469);
    let r = i64x2::new(55834574861, 55834574861);

    assert_eq!(r, transmute(lsx_vmaxi_w::<13>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaxi_d() {
    let a = i64x2::new(-490958606840895025, -602287987736508723);
    let r = i64x2::new(-5, -5);

    assert_eq!(r, transmute(lsx_vmaxi_d::<-5>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmax_bu() {
    let a = u8x16::new(
        22, 96, 70, 57, 83, 248, 184, 163, 4, 150, 223, 247, 226, 242, 18, 63,
    );
    let b = u8x16::new(
        13, 251, 236, 121, 148, 91, 24, 176, 232, 197, 195, 34, 31, 120, 173, 27,
    );
    let r = i64x2::new(-5712542810735052010, 4588590651995571688);

    assert_eq!(r, transmute(lsx_vmax_bu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmax_hu() {
    let a = u16x8::new(1178, 52364, 32269, 22619, 17388, 4159, 51894, 12662);
    let b = u16x8::new(61508, 27224, 11696, 15294, 30725, 4809, 55995, 24012);
    let r = i64x2::new(6366821095949791300, 6759017637785204741);

    assert_eq!(r, transmute(lsx_vmax_hu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmax_wu() {
    let a = u32x4::new(2081333956, 40837464, 1440470019, 1657093799);
    let b = u32x4::new(2856502284, 546582019, 3814541188, 2370198139);
    let r = i64x2::new(2347551899043152908, -8266820577849948284);

    assert_eq!(r, transmute(lsx_vmax_wu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmax_du() {
    let a = u64x2::new(17105634039018730835, 11926654155810942548);
    let b = u64x2::new(15559502733477870114, 3537017767853389449);
    let r = i64x2::new(-1341110034690820781, -6520089917898609068);

    assert_eq!(r, transmute(lsx_vmax_du(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaxi_bu() {
    let a = u8x16::new(
        216, 225, 158, 238, 152, 8, 124, 241, 175, 62, 154, 175, 216, 127, 235, 143,
    );
    let r = i64x2::new(-1045930669804428840, -8076220938123067729);

    assert_eq!(r, transmute(lsx_vmaxi_bu::<27>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaxi_hu() {
    let a = u16x8::new(56394, 18974, 59, 64239, 15178, 38205, 20044, 21066);
    let r = i64x2::new(-365072790147113910, 5929637950214978378);

    assert_eq!(r, transmute(lsx_vmaxi_hu::<23>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaxi_wu() {
    let a = u32x4::new(2234002286, 3837532269, 3218694441, 2956128392);
    let r = i64x2::new(-1964668478775874706, -5750269304073789143);

    assert_eq!(r, transmute(lsx_vmaxi_wu::<15>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaxi_du() {
    let a = u64x2::new(3145066433415682744, 697260191203805367);
    let r = i64x2::new(3145066433415682744, 697260191203805367);

    assert_eq!(r, transmute(lsx_vmaxi_du::<15>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmin_b() {
    let a = i8x16::new(
        -18, -126, -77, 105, 18, -106, -12, 89, 93, 22, -51, -103, -63, -106, -23, -125,
    );
    let b = i8x16::new(
        -10, 83, 19, -119, -1, 95, 11, 25, -11, 38, -28, -23, -36, -104, 110, 0,
    );
    let r = i64x2::new(1870285769536668398, -8941449826914199819);

    assert_eq!(r, transmute(lsx_vmin_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmin_h() {
    let a = i16x8::new(7767, 30288, -1525, 24469, 16179, 7042, 6326, 21055);
    let b = i16x8::new(-5519, 15267, -28304, -5842, 32145, 6582, -9646, -24918);
    let r = i64x2::new(-1644216902720689551, -7013553423522578637);

    assert_eq!(r, transmute(lsx_vmin_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmin_w() {
    let a = i32x4::new(280954204, 1916591882, 1901481995, 787566518);
    let b = i32x4::new(-425011290, -2104111279, 175390640, 571448257);
    let r = i64x2::new(-9037089126579775578, 2454351575346593712);

    assert_eq!(r, transmute(lsx_vmin_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmin_d() {
    let a = i64x2::new(5262417572890363865, 5296071757031183187);
    let b = i64x2::new(7269804448576860985, -2384075780126369706);
    let r = i64x2::new(5262417572890363865, -2384075780126369706);

    assert_eq!(r, transmute(lsx_vmin_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmini_b() {
    let a = i8x16::new(
        -20, 19, 89, -115, 65, 94, -124, -17, 36, -127, -101, -123, -122, -62, 44, 121,
    );
    let r = i64x2::new(-1187557278141451540, -940475489144045070);

    assert_eq!(r, transmute(lsx_vmini_b::<-14>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmini_h() {
    let a = i16x8::new(26119, -26421, -26720, 11534, 11181, -13024, -9525, -1565);
    let r = i64x2::new(-677708916064259, -440267769697468419);

    assert_eq!(r, transmute(lsx_vmini_h::<-3>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmini_w() {
    let a = i32x4::new(1937226480, -56354461, -210581139, 118641668);
    let r = i64x2::new(-242040566978707451, 25559222637);

    assert_eq!(r, transmute(lsx_vmini_w::<5>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmini_d() {
    let a = i64x2::new(-6839357499730806877, 2982085289136510651);
    let r = i64x2::new(-6839357499730806877, 11);

    assert_eq!(r, transmute(lsx_vmini_d::<11>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmin_bu() {
    let a = u8x16::new(
        72, 253, 194, 62, 100, 41, 53, 50, 53, 249, 47, 215, 113, 227, 189, 66,
    );
    let b = u8x16::new(
        20, 165, 214, 231, 201, 17, 81, 203, 41, 209, 98, 88, 135, 118, 100, 83,
    );
    let r = i64x2::new(3617816997909406996, 4784078933357220137);

    assert_eq!(r, transmute(lsx_vmin_bu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmin_hu() {
    let a = u16x8::new(45665, 56395, 48109, 47478, 46813, 59058, 42125, 32550);
    let b = u16x8::new(30424, 14541, 7654, 46014, 42452, 14971, 14903, 13871);
    let r = i64x2::new(-5494921620712753448, 3904403410832303572);

    assert_eq!(r, transmute(lsx_vmin_hu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmin_wu() {
    let a = u32x4::new(1809171870, 3212127932, 1131140001, 2157144340);
    let b = u32x4::new(1456829356, 2264966310, 1587887390, 645429404);
    let r = i64x2::new(-8718787844260924500, 2772098183187911585);

    assert_eq!(r, transmute(lsx_vmin_wu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmin_du() {
    let a = u64x2::new(6641707046382446478, 5750385968612732680);
    let b = u64x2::new(15079551366517035256, 13891052596545854864);
    let r = i64x2::new(6641707046382446478, 5750385968612732680);

    assert_eq!(r, transmute(lsx_vmin_du(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmini_bu() {
    let a = u8x16::new(
        14, 244, 217, 183, 206, 234, 5, 185, 152, 22, 4, 35, 30, 177, 252, 137,
    );
    let r = i64x2::new(361700864190383365, 361700864190317829);

    assert_eq!(r, transmute(lsx_vmini_bu::<5>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmini_hu() {
    let a = u16x8::new(51791, 41830, 16737, 31634, 36341, 58491, 48701, 8690);
    let r = i64x2::new(5066626891382802, 5066626891382802);

    assert_eq!(r, transmute(lsx_vmini_hu::<18>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmini_wu() {
    let a = u32x4::new(1158888991, 2639721369, 556001789, 2902942998);
    let r = i64x2::new(77309411346, 77309411346);

    assert_eq!(r, transmute(lsx_vmini_wu::<18>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmini_du() {
    let a = u64x2::new(17903595768445663391, 13119300660970895532);
    let r = i64x2::new(13, 13);

    assert_eq!(r, transmute(lsx_vmini_du::<13>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vseq_b() {
    let a = i8x16::new(
        8, 73, 39, 20, 64, -98, -64, 83, 32, 84, -121, 9, -45, -118, -26, 100,
    );
    let b = i8x16::new(
        -90, -2, -77, -76, -19, 48, 91, 31, 65, -29, -112, -7, 77, 98, -126, 5,
    );
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vseq_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vseq_h() {
    let a = i16x8::new(7490, 32190, -24684, 16245, -18425, -12556, 19179, -23230);
    let b = i16x8::new(-7387, -24074, 15709, -4629, 30465, -9504, -21403, -30287);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vseq_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vseq_w() {
    let a = i32x4::new(-364333737, 833593451, -1047433707, 1224903962);
    let b = i32x4::new(-493722413, -522973881, -1254416384, -884207273);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vseq_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vseq_d() {
    let a = i64x2::new(8059130761383772313, -728251064129355704);
    let b = i64x2::new(3023654898382436999, 1783520577741396523);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vseq_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vseqi_b() {
    let a = i8x16::new(
        114, -39, -58, -47, -46, 68, 126, -41, 50, -24, 109, 120, -81, -22, 86, 2,
    );
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vseqi_b::<12>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vseqi_h() {
    let a = i16x8::new(-3205, 25452, 20774, 22065, -8424, 16590, -15971, -14154);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vseqi_h::<-1>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vseqi_w() {
    let a = i32x4::new(199798215, -798304779, -1812193878, -1830438161);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vseqi_w::<11>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vseqi_d() {
    let a = i64x2::new(-7376858177879278972, 1947027764115386661);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vseqi_d::<3>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslti_b() {
    let a = i8x16::new(
        45, 70, 62, 83, 116, -29, -34, -91, 96, 48, 109, 92, -18, 93, 14, 22,
    );
    let r = i64x2::new(-1099511627776, 1095216660480);

    assert_eq!(r, transmute(lsx_vslti_b::<-4>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslt_b() {
    let a = i8x16::new(
        -68, 126, 28, -97, -24, 118, 61, -9, 5, 115, -122, 5, -40, 107, -98, -93,
    );
    let b = i8x16::new(
        22, 124, 33, 93, 0, -81, -62, 63, 1, 35, -64, 23, 61, 9, -56, 89,
    );
    let r = i64x2::new(-72056494526365441, -280375465148416);

    assert_eq!(r, transmute(lsx_vslt_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslt_h() {
    let a = i16x8::new(32283, 16403, -32598, 8049, -10290, 21116, 23894, 5619);
    let b = i16x8::new(-10624, 12762, 31216, 13253, 2299, -12591, -8652, -22348);
    let r = i64x2::new(-4294967296, 65535);

    assert_eq!(r, transmute(lsx_vslt_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslt_w() {
    let a = i32x4::new(-158999818, -1928813163, -140040541, 494178107);
    let b = i32x4::new(-1849021639, -756143028, 54274044, 646446450);
    let r = i64x2::new(-4294967296, -1);

    assert_eq!(r, transmute(lsx_vslt_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslt_d() {
    let a = i64x2::new(-179055155347449719, 6182805737835801255);
    let b = i64x2::new(1481173131774551907, 270656941607020532);
    let r = i64x2::new(-1, 0);

    assert_eq!(r, transmute(lsx_vslt_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslti_h() {
    let a = i16x8::new(-8902, 5527, 17224, -27356, 4424, 28839, 29975, 18805);
    let r = i64x2::new(-281474976645121, 0);

    assert_eq!(r, transmute(lsx_vslti_h::<14>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslti_w() {
    let a = i32x4::new(995282502, -1964668207, -996118772, 1812234755);
    let r = i64x2::new(-4294967296, 4294967295);

    assert_eq!(r, transmute(lsx_vslti_w::<14>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslti_d() {
    let a = i64x2::new(1441753618400573134, 3878439049744730841);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vslti_d::<14>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslt_bu() {
    let a = u8x16::new(
        55, 192, 87, 242, 253, 133, 53, 76, 135, 6, 39, 64, 82, 182, 147, 19,
    );
    let b = u8x16::new(
        108, 77, 229, 137, 242, 115, 152, 252, 99, 101, 44, 100, 58, 120, 101, 22,
    );
    let r = i64x2::new(-281474959998721, -72057589742960896);

    assert_eq!(r, transmute(lsx_vslt_bu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslt_hu() {
    let a = u16x8::new(16382, 2642, 8944, 48121, 7472, 49176, 63264, 1135);
    let b = u16x8::new(513, 13075, 20319, 44422, 12609, 18638, 20227, 21354);
    let r = i64x2::new(281474976645120, -281474976645121);

    assert_eq!(r, transmute(lsx_vslt_hu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslt_wu() {
    let a = u32x4::new(137339688, 2061001419, 2322333619, 2113106148);
    let b = u32x4::new(1402243125, 1129899238, 2591537060, 4152171743);
    let r = i64x2::new(4294967295, -1);

    assert_eq!(r, transmute(lsx_vslt_wu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslt_du() {
    let a = u64x2::new(15914553432791856307, 11132190561956652500);
    let b = u64x2::new(835355141719377733, 10472626544222695938);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vslt_du(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslti_bu() {
    let a = u8x16::new(
        215, 70, 65, 148, 249, 56, 59, 18, 118, 56, 250, 53, 144, 189, 98, 56,
    );
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vslti_bu::<7>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslti_hu() {
    let a = u16x8::new(60550, 12178, 30950, 44771, 25514, 35987, 55940, 21614);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vslti_hu::<2>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslti_wu() {
    let a = u32x4::new(912580668, 18660032, 3405726641, 4033549497);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vslti_wu::<8>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslti_du() {
    let a = u64x2::new(17196150830761730262, 5893061291971214149);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vslti_du::<14>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsle_b() {
    let a = i8x16::new(
        16, 13, 47, 41, 9, -73, 92, 108, -77, -106, -115, -20, 107, -101, -54, 16,
    );
    let b = i8x16::new(
        71, 43, 24, 28, 83, 69, -109, -33, 81, 71, -126, -61, -45, -11, -105, -70,
    );
    let r = i64x2::new(281470681808895, 280375465148415);

    assert_eq!(r, transmute(lsx_vsle_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsle_h() {
    let a = i16x8::new(15130, 12644, -27298, 13979, 28696, -28425, 23806, -20696);
    let b = i16x8::new(-30602, -9535, 10944, 3343, -1093, 6600, -19453, -4561);
    let r = i64x2::new(281470681743360, -281470681808896);

    assert_eq!(r, transmute(lsx_vsle_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsle_w() {
    let a = i32x4::new(-549852719, 335768045, 1882235130, 603655976);
    let b = i32x4::new(-1810853975, 2021418524, 215198844, 1124361386);
    let r = i64x2::new(-4294967296, -4294967296);

    assert_eq!(r, transmute(lsx_vsle_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsle_d() {
    let a = i64x2::new(-5807954019703375704, 7802006580674332206);
    let b = i64x2::new(71694374951002423, -4307912969104303925);
    let r = i64x2::new(-1, 0);

    assert_eq!(r, transmute(lsx_vsle_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslei_b() {
    let a = i8x16::new(
        22, -8, 10, 55, 103, -103, -106, 30, 54, 82, 29, 44, 75, -9, 36, 111,
    );
    let r = i64x2::new(72056494526365440, 280375465082880);

    assert_eq!(r, transmute(lsx_vslei_b::<3>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslei_h() {
    let a = i16x8::new(31276, -16628, -30006, -20587, 2104, -30062, 18261, -6449);
    let r = i64x2::new(-65536, -281470681808896);

    assert_eq!(r, transmute(lsx_vslei_h::<-3>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslei_w() {
    let a = i32x4::new(-1890390435, 1289536678, 1490122113, 2120063492);
    let r = i64x2::new(4294967295, 0);

    assert_eq!(r, transmute(lsx_vslei_w::<-16>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslei_d() {
    let a = i64x2::new(-123539898448811963, 8007480165241051883);
    let r = i64x2::new(-1, 0);

    assert_eq!(r, transmute(lsx_vslei_d::<8>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsle_bu() {
    let a = u8x16::new(
        156, 210, 61, 51, 143, 107, 237, 69, 241, 117, 66, 79, 161, 68, 22, 152,
    );
    let b = u8x16::new(
        83, 68, 27, 36, 209, 74, 204, 32, 123, 97, 44, 82, 238, 202, 133, 107,
    );
    let r = i64x2::new(1095216660480, 72057594021150720);

    assert_eq!(r, transmute(lsx_vsle_bu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsle_hu() {
    let a = u16x8::new(57583, 52549, 12485, 59674, 7283, 26602, 6409, 58628);
    let b = u16x8::new(50529, 35111, 24746, 62465, 21587, 30574, 11054, 11653);
    let r = i64x2::new(-4294967296, 281474976710655);

    assert_eq!(r, transmute(lsx_vsle_hu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsle_wu() {
    let a = u32x4::new(3325048208, 3863618944, 2967312103, 2626474550);
    let b = u32x4::new(1321018603, 1091195011, 3525236625, 4061062671);
    let r = i64x2::new(0, -1);

    assert_eq!(r, transmute(lsx_vsle_wu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsle_du() {
    let a = u64x2::new(17131200460153340378, 17148253643287276161);
    let b = u64x2::new(16044633718831874991, 3531311371811276914);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vsle_du(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslei_bu() {
    let a = u8x16::new(
        33, 181, 170, 160, 192, 237, 16, 175, 82, 65, 186, 46, 143, 9, 37, 35,
    );
    let r = i64x2::new(71776119061217280, 280375465082880);

    assert_eq!(r, transmute(lsx_vslei_bu::<18>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslei_hu() {
    let a = u16x8::new(1430, 10053, 35528, 28458, 2394, 22098, 40236, 20853);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vslei_hu::<10>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslei_wu() {
    let a = u32x4::new(3289026584, 3653636092, 2919866047, 2895662832);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vslei_wu::<2>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vslei_du() {
    let a = u64x2::new(17462377852989253439, 17741928456729041079);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vslei_du::<12>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsat_b() {
    let a = i8x16::new(
        -66, 2, -76, 126, 9, -44, -37, -42, 8, 68, -72, 10, 113, 70, 58, 44,
    );
    let r = i64x2::new(-2964542792447819074, 3186937137643144200);

    assert_eq!(r, transmute(lsx_vsat_b::<7>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsat_h() {
    let a = i16x8::new(-22234, -8008, -23350, 13768, 26313, -27447, -3569, 6025);
    let r = i64x2::new(576451960371214336, 576451960371152895);

    assert_eq!(r, transmute(lsx_vsat_h::<11>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsat_w() {
    let a = i32x4::new(-84179653, 874415975, 1823119516, 1667850968);
    let r = i64x2::new(137438953440, 133143986207);

    assert_eq!(r, transmute(lsx_vsat_w::<5>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsat_d() {
    let a = i64x2::new(6859869867233872152, 2514172105675226457);
    let r = i64x2::new(262143, 262143);

    assert_eq!(r, transmute(lsx_vsat_d::<18>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsat_bu() {
    let a = u8x16::new(
        119, 190, 12, 39, 41, 110, 238, 29, 14, 135, 54, 90, 36, 89, 72, 91,
    );
    let r = i64x2::new(2125538672170008439, 6577605268441825038);

    assert_eq!(r, transmute(lsx_vsat_bu::<6>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsat_hu() {
    let a = u16x8::new(36681, 34219, 6160, 8687, 4544, 20195, 35034, 916);
    let r = i64x2::new(287953294993589247, 257835472485549055);

    assert_eq!(r, transmute(lsx_vsat_hu::<9>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsat_wu() {
    let a = u32x4::new(1758000759, 4138051566, 2705324001, 3927640324);
    let r = i64x2::new(70364449226751, 70364449226751);

    assert_eq!(r, transmute(lsx_vsat_wu::<13>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsat_du() {
    let a = u64x2::new(1953136817312581670, 2606878300382729363);
    let r = i64x2::new(9007199254740991, 9007199254740991);

    assert_eq!(r, transmute(lsx_vsat_du::<52>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vadda_b() {
    let a = i8x16::new(
        -44, -56, -103, -51, 118, -127, -39, -96, -49, 75, -110, 35, 123, -61, 57, 104,
    );
    let b = i8x16::new(
        79, 88, -93, 36, 117, -15, -81, -18, -117, -47, -13, 83, -31, -61, 60, 14,
    );
    let r = i64x2::new(8248499858970022011, 8535863472581999270);

    assert_eq!(r, transmute(lsx_vadda_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vadda_h() {
    let a = i16x8::new(15992, -5603, -27115, -15673, 11461, -31471, -31137, -2291);
    let b = i16x8::new(-21543, 21720, 14529, -19143, -28953, 13450, 8037, 29413);
    let r = i64x2::new(-8646732423142600033, 8924050915627474398);

    assert_eq!(r, transmute(lsx_vadda_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vadda_w() {
    let a = i32x4::new(1188987464, -1693707744, -1561184997, -104072194);
    let b = i32x4::new(287041349, 249467792, 312776520, 1314435078);
    let r = i64x2::new(8345875378983299469, 6092442344252138029);

    assert_eq!(r, transmute(lsx_vadda_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vadda_d() {
    let a = i64x2::new(1747309060022550268, -6715694127559156035);
    let b = i64x2::new(-4324432602362661920, 6402427893748093984);
    let r = i64x2::new(6071741662385212188, -5328622052402301597);

    assert_eq!(r, transmute(lsx_vadda_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsadd_b() {
    let a = i8x16::new(
        6, -114, -40, 76, -8, 4, -110, -105, -104, 86, -27, 68, -102, 108, 113, 76,
    );
    let b = i8x16::new(
        -47, 102, 105, 84, -127, 70, -116, 57, 66, 47, 74, -35, 61, -85, 48, -50,
    );
    let r = i64x2::new(-3422653801050278697, 1909270979770548186);

    assert_eq!(r, transmute(lsx_vsadd_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsadd_h() {
    let a = i16x8::new(-25724, -16509, -25895, 31488, -18727, 16765, 3340, 21218);
    let b = i16x8::new(26970, 17131, 15547, -7614, -8479, 22338, 3567, -22299);
    let r = i64x2::new(6720170624686097630, -304244782337649222);

    assert_eq!(r, transmute(lsx_vsadd_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsadd_w() {
    let a = i32x4::new(-1981320133, -1751087788, 1176481176, 253883202);
    let b = i32x4::new(-1026388582, 222487110, 501504960, -1863994162);
    let r = i64x2::new(-6565289918505943040, -6915373914453178024);

    assert_eq!(r, transmute(lsx_vsadd_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsadd_d() {
    let a = i64x2::new(-1967787987610391555, -8103697759704177767);
    let b = i64x2::new(-6599608819082608284, -5088169537193133686);
    let r = i64x2::new(-8567396806692999839, -9223372036854775808);

    assert_eq!(r, transmute(lsx_vsadd_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsadd_bu() {
    let a = u8x16::new(
        182, 156, 225, 235, 23, 111, 224, 152, 158, 254, 143, 58, 230, 188, 119, 239,
    );
    let b = u8x16::new(
        40, 219, 72, 211, 12, 37, 59, 28, 206, 173, 87, 21, 125, 229, 110, 102,
    );
    let r = i64x2::new(-5404438145481572386, -7318352348905473);

    assert_eq!(r, transmute(lsx_vsadd_bu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsadd_hu() {
    let a = u16x8::new(52962, 42889, 37893, 55695, 51804, 38647, 13774, 40745);
    let b = u16x8::new(31219, 59227, 25607, 62798, 18845, 3238, 19902, 24978);
    let r = i64x2::new(-8740258447361, -136834913009665);

    assert_eq!(r, transmute(lsx_vsadd_hu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsadd_wu() {
    let a = u32x4::new(1617769210, 1445524000, 4168062781, 912440538);
    let b = u32x4::new(3676524021, 3894343575, 904432536, 1616820031);
    let r = i64x2::new(-1, -7583652642497232897);

    assert_eq!(r, transmute(lsx_vsadd_wu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsadd_du() {
    let a = u64x2::new(3740778533337193809, 14274264382641271168);
    let b = u64x2::new(11054638512585704882, 3549000132135395099);
    let r = i64x2::new(-3651327027786652925, -623479558932885349);

    assert_eq!(r, transmute(lsx_vsadd_du(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vavg_b() {
    let a = i8x16::new(
        117, 127, 54, 98, -91, 42, 42, 76, 29, 63, -21, 26, -77, -7, -81, 78,
    );
    let b = i8x16::new(
        30, 62, -76, -20, 127, 89, -99, -82, 69, -114, 84, 80, -78, -102, -107, 43,
    );
    let r = i64x2::new(-152206416164856247, 4369276355735447089);

    assert_eq!(r, transmute(lsx_vavg_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vavg_h() {
    let a = i16x8::new(-12604, -917, -12088, 13367, -2577, -1073, 1365, -25654);
    let b = i16x8::new(-3088, -25854, -32552, -8417, 7808, -12495, 22032, -5168);
    let r = i64x2::new(696836182083297626, -4337760619710117321);

    assert_eq!(r, transmute(lsx_vavg_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vavg_w() {
    let a = i32x4::new(826230751, 1801449269, -284345024, 1777295732);
    let b = i32x4::new(-324844828, -1580060766, -1909832882, 328273785);
    let r = i64x2::new(475428188150908257, 4521676108535152711);

    assert_eq!(r, transmute(lsx_vavg_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vavg_d() {
    let a = i64x2::new(1486723108337487211, 6178549804180384276);
    let b = i64x2::new(3169904420607189220, 5159962511251707672);
    let r = i64x2::new(2328313764472338215, 5669256157716045974);

    assert_eq!(r, transmute(lsx_vavg_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vavg_bu() {
    let a = u8x16::new(
        84, 85, 64, 60, 241, 96, 145, 145, 51, 253, 205, 150, 135, 87, 248, 55,
    );
    let b = u8x16::new(
        179, 216, 158, 135, 196, 75, 59, 209, 8, 58, 142, 152, 16, 220, 199, 21,
    );
    let r = i64x2::new(-5663745084945885565, 2801126043194071837);

    assert_eq!(r, transmute(lsx_vavg_bu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vavg_hu() {
    let a = u16x8::new(46978, 53346, 32276, 58377, 57638, 42860, 43999, 59924);
    let b = u16x8::new(44835, 36733, 12115, 42874, 4819, 12201, 27397, 25394);
    let r = i64x2::new(-4196978047981735086, -6439149718662907396);

    assert_eq!(r, transmute(lsx_vavg_hu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vavg_wu() {
    let a = u32x4::new(529045804, 31575520, 1599127613, 3465214369);
    let b = u32x4::new(160886383, 26081142, 459122380, 2523086630);
    let r = i64x2::new(123816739188229069, -5586965600173345916);

    assert_eq!(r, transmute(lsx_vavg_wu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vavg_du() {
    let a = u64x2::new(11603952465622489487, 9916150703735650033);
    let b = u64x2::new(9749063966076740681, 5963120178993456389);
    let r = i64x2::new(-7770235857859936532, 7939635441364553211);

    assert_eq!(r, transmute(lsx_vavg_du(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vavgr_b() {
    let a = i8x16::new(
        42, -6, 89, -102, -107, 103, 13, -3, -19, -93, 0, 0, -17, 70, 54, 86,
    );
    let b = i8x16::new(
        8, -32, -122, 22, -94, 44, 58, 54, -26, -34, -21, 27, -111, -96, -68, -122,
    );
    let r = i64x2::new(1883712581662731545, -1226681417271426582);

    assert_eq!(r, transmute(lsx_vavgr_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vavgr_h() {
    let a = i16x8::new(-6008, 3940, -4691, -4052, 15265, -7180, 976, 11656);
    let b = i16x8::new(-9758, -8332, 20577, 31066, 31120, 14788, -22323, 16722);
    let r = i64x2::new(3801916629507170613, 3994084079587580569);

    assert_eq!(r, transmute(lsx_vavgr_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vavgr_w() {
    let a = i32x4::new(-518881442, 2037406651, -1244322310, -1948025633);
    let b = i32x4::new(1278058715, -155858446, -195547847, -750518746);
    let r = i64x2::new(4040594005688324125, -5795079921582298726);

    assert_eq!(r, transmute(lsx_vavgr_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vavgr_d() {
    let a = i64x2::new(-1958143381023430514, 3633380184275298119);
    let b = i64x2::new(8758126674980055299, -7441643514470614533);
    let r = i64x2::new(3399991646978312393, -1904131665097658207);

    assert_eq!(r, transmute(lsx_vavgr_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vavgr_bu() {
    let a = u8x16::new(
        205, 114, 125, 237, 6, 194, 197, 217, 10, 191, 130, 30, 247, 116, 199, 100,
    );
    let b = u8x16::new(
        6, 139, 195, 209, 115, 27, 109, 34, 91, 48, 166, 147, 170, 83, 9, 65,
    );
    let r = i64x2::new(9122444831751176042, 6010164553039771699);

    assert_eq!(r, transmute(lsx_vavgr_bu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vavgr_hu() {
    let a = u16x8::new(49326, 55416, 46414, 26192, 61759, 37293, 22943, 26741);
    let b = u16x8::new(26111, 34713, 61420, 23702, 29204, 9543, 62786, 7043);
    let r = i64x2::new(7022187818705851223, 4754859411904311722);

    assert_eq!(r, transmute(lsx_vavgr_hu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vavgr_wu() {
    let a = u32x4::new(3560278529, 2406185766, 3420917939, 1379681517);
    let b = u32x4::new(1930150361, 3668628165, 2983921396, 2410913126);
    let r = i64x2::new(-5401180487351753235, 8140240017388800980);

    assert_eq!(r, transmute(lsx_vavgr_wu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vavgr_du() {
    let a = u64x2::new(3442342130569215862, 4810216499730807927);
    let b = u64x2::new(8650759135311802962, 11380630663742852932);
    let r = i64x2::new(6046550632940509412, 8095423581736830430);

    assert_eq!(r, transmute(lsx_vavgr_du(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssub_b() {
    let a = i8x16::new(
        49, 58, 94, 93, 7, 40, -34, 27, 75, -67, -71, 2, -117, -22, 78, -78,
    );
    let b = i8x16::new(
        -104, 71, -79, -113, 21, 34, 36, 19, 92, 32, -77, 91, 28, -43, -69, 62,
    );
    let r = i64x2::new(628822736562549631, -9187601072510296593);

    assert_eq!(r, transmute(lsx_vssub_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssub_h() {
    let a = i16x8::new(14676, -4176, 31759, -22564, 6643, 20831, 15260, 18518);
    let b = i16x8::new(-26027, 6118, -13204, 25080, 12458, 8441, 24701, 11617);
    let r = i64x2::new(-9223231300041015297, 1942699741282756937);

    assert_eq!(r, transmute(lsx_vssub_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssub_w() {
    let a = i32x4::new(-359085176, -924784873, 1280567100, 1138686008);
    let b = i32x4::new(-1808829767, 2144666490, 146236682, 1180114488);
    let r = i64x2::new(-9223372035405031217, -177933965588659662);

    assert_eq!(r, transmute(lsx_vssub_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssub_d() {
    let a = i64x2::new(628092957162650618, 1527439654680677883);
    let b = i64x2::new(-2293337525465880409, 5736255249834646932);
    let r = i64x2::new(2921430482628531027, -4208815595153969049);

    assert_eq!(r, transmute(lsx_vssub_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssub_bu() {
    let a = u8x16::new(
        198, 146, 80, 65, 122, 45, 61, 106, 212, 129, 170, 111, 183, 102, 130, 148,
    );
    let b = u8x16::new(
        16, 110, 145, 170, 113, 220, 82, 86, 9, 255, 200, 230, 204, 22, 213, 203,
    );
    let r = i64x2::new(1441151919413273782, 87960930222283);

    assert_eq!(r, transmute(lsx_vssub_bu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssub_hu() {
    let a = u16x8::new(62355, 31259, 41090, 62278, 449, 36606, 38644, 57485);
    let b = u16x8::new(50468, 33060, 15257, 59071, 59343, 21993, 42978, 20097);
    let r = i64x2::new(902801202201243247, -7922957643493867520);

    assert_eq!(r, transmute(lsx_vssub_hu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssub_wu() {
    let a = u32x4::new(360162968, 3504892941, 1150347916, 2195977376);
    let b = u32x4::new(31483972, 3489479082, 152079374, 1875131600);
    let r = i64x2::new(66202020638834260, 1378022115978010238);

    assert_eq!(r, transmute(lsx_vssub_wu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssub_du() {
    let a = u64x2::new(14887776146288736271, 417684393846230822);
    let b = u64x2::new(6460869225596371206, 16765308520486969885);
    let r = i64x2::new(8426906920692365065, 0);

    assert_eq!(r, transmute(lsx_vssub_du(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vabsd_b() {
    let a = i8x16::new(
        -80, -35, -110, -126, -9, -18, -111, -50, -68, 115, -53, 79, -35, 102, -85, 68,
    );
    let b = i8x16::new(
        85, -87, -91, 4, -102, 47, 70, 8, -16, 86, -14, -127, 2, -58, 10, 39,
    );
    let r = i64x2::new(4230359294854509733, 2116586434120326452);

    assert_eq!(r, transmute(lsx_vabsd_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vabsd_h() {
    let a = i16x8::new(-9487, 3116, 31071, -3514, -4374, 29502, 15788, 8887);
    let b = i16x8::new(9346, 27961, 21592, 10762, -6831, 17219, 14968, -1750);
    let r = i64x2::new(4018377481144584593, 2994052849949411737);

    assert_eq!(r, transmute(lsx_vabsd_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vabsd_w() {
    let a = i32x4::new(1772435833, -142335623, -905419863, -1391379125);
    let b = i32x4::new(-638463360, -1154268425, 818053243, -1766966029);
    let r = i64x2::new(4346218292750542585, 1613133471209364690);

    assert_eq!(r, transmute(lsx_vabsd_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vabsd_d() {
    let a = i64x2::new(-1345697660428932390, -6981332546532147421);
    let b = i64x2::new(-8533946706796471089, 1165272962517390961);
    let r = i64x2::new(7188249046367538699, 8146605509049538382);

    assert_eq!(r, transmute(lsx_vabsd_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vabsd_bu() {
    let a = u8x16::new(
        3, 31, 230, 199, 201, 67, 112, 189, 15, 214, 56, 113, 214, 23, 217, 54,
    );
    let b = u8x16::new(
        207, 196, 133, 201, 150, 94, 74, 221, 222, 61, 222, 248, 105, 208, 154, 128,
    );
    let r = i64x2::new(2316568964225934796, 5350198762417854927);

    assert_eq!(r, transmute(lsx_vabsd_bu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vabsd_hu() {
    let a = u16x8::new(30314, 20737, 52964, 57347, 14004, 37245, 9170, 22466);
    let b = u16x8::new(42102, 40052, 6807, 16289, 29686, 38061, 42843, 26642);
    let r = i64x2::new(-6889746235852116468, 1175584127230950722);

    assert_eq!(r, transmute(lsx_vabsd_hu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vabsd_wu() {
    let a = u32x4::new(1481954749, 4094293310, 3199531334, 4211151920);
    let b = u32x4::new(3008439409, 976530727, 1726048801, 4235308512);
    let r = i64x2::new(-5056055741505581388, 103751774096297765);

    assert_eq!(r, transmute(lsx_vabsd_wu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vabsd_du() {
    let a = u64x2::new(14212221485552223583, 1471016340493959617);
    let b = u64x2::new(305704565845198935, 18327726360649467511);
    let r = i64x2::new(-4540227154002526968, -1590034053554043722);

    assert_eq!(r, transmute(lsx_vabsd_du(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmul_b() {
    let a = i8x16::new(
        -108, -77, -99, -81, 97, 59, -58, 100, 104, -89, -58, -96, -25, 125, 127, -61,
    );
    let b = i8x16::new(
        64, 109, -119, -124, -55, -11, -90, -123, 72, -18, 83, 46, 102, -25, -11, 27,
    );
    let r = i64x2::new(-836412611799730432, -7959044669412588992);

    assert_eq!(r, transmute(lsx_vmul_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmul_h() {
    let a = i16x8::new(20255, 19041, 15158, 5077, -29421, -8508, 6583, -968);
    let b = i16x8::new(-18582, -25667, 17674, 8424, -17121, -21798, 28934, -353);
    let r = i64x2::new(-7419436171490628650, 3947512047518358605);

    assert_eq!(r, transmute(lsx_vmul_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmul_w() {
    let a = i32x4::new(1875532791, -2038975148, 754073945, 1245315915);
    let b = i32x4::new(1754730718, 782084571, 894216679, -1895747372);
    let r = i64x2::new(6602438528086061106, 4680306660704041039);

    assert_eq!(r, transmute(lsx_vmul_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmul_d() {
    let a = i64x2::new(-4093110041189429887, 5371368149814248867);
    let b = i64x2::new(8096709215426138432, -5454415917204378153);
    let r = i64x2::new(-1062747544199352000, -649255846668983579);

    assert_eq!(r, transmute(lsx_vmul_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmadd_b() {
    let a = i8x16::new(
        60, 90, -59, 50, 52, 30, -124, 62, -71, -71, -38, 22, 6, -18, 93, 102,
    );
    let b = i8x16::new(
        22, 41, -112, 44, -93, -82, 11, -47, 37, -120, -108, 33, -66, 27, -74, -2,
    );
    let c = i8x16::new(
        103, 59, 65, -2, -55, 98, -11, 85, 84, 50, -17, 14, -19, 120, 7, -90,
    );
    let r = i64x2::new(-6698055306094195434, 1898151712142019037);

    assert_eq!(
        r,
        transmute(lsx_vmadd_b(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmadd_h() {
    let a = i16x8::new(24257, 11879, -5695, -12734, -31748, 30664, 11820, 3259);
    let b = i16x8::new(23734, 11732, -14134, -26857, 30756, 2629, 25687, 15749);
    let c = i16x8::new(-9000, -804, 10411, 17571, -4985, -22809, -5536, -1762);
    let r = i64x2::new(2154858825190408273, -6966693911367840008);

    assert_eq!(
        r,
        transmute(lsx_vmadd_h(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmadd_w() {
    let a = i32x4::new(1344709991, 1633778942, 1825268167, 917193207);
    let b = i32x4::new(147354288, -1478483633, -941638228, -173023515);
    let c = i32x4::new(-1301057792, -1104623642, -1440212635, -8186971);
    let r = i64x2::new(4970798576846304615, -3981205637140381021);

    assert_eq!(
        r,
        transmute(lsx_vmadd_w(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmadd_d() {
    let a = i64x2::new(-7021558423493045864, 7607197079929138141);
    let b = i64x2::new(-7461017148544541027, -326746346508808472);
    let c = i64x2::new(9019083511238971943, 8084580083589700502);
    let r = i64x2::new(-7790478971542305405, -5909066061947936819);

    assert_eq!(
        r,
        transmute(lsx_vmadd_d(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmsub_b() {
    let a = i8x16::new(
        -114, -46, 82, -75, -22, 31, 79, 84, -108, -13, -40, -121, -2, -20, 75, -35,
    );
    let b = i8x16::new(
        -29, 61, -62, 87, -22, 53, 51, 24, -27, -74, 119, -20, 21, 5, 14, -92,
    );
    let c = i8x16::new(
        -57, 111, 112, -66, 100, -31, -70, -71, 92, 63, 108, 61, -115, 17, -75, 16,
    );
    let r = i64x2::new(-269782211120439527, -7105106341430810296);

    assert_eq!(
        r,
        transmute(lsx_vmsub_b(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmsub_h() {
    let a = i16x8::new(28727, 27408, -23829, -25297, 24892, 31830, -2674, -17919);
    let b = i16x8::new(6329, 13060, 18913, 18407, 28125, -26009, -14135, 22627);
    let c = i16x8::new(26144, 29029, 6084, 10072, 21090, -4197, 21706, -19485);
    let r = i64x2::new(-5420122113954766057, 2393824782223771810);

    assert_eq!(
        r,
        transmute(lsx_vmsub_h(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmsub_w() {
    let a = i32x4::new(385413537, 143148625, 1902013465, -1637986171);
    let b = i32x4::new(-1124183308, 1253368192, 1310051041, -750553442);
    let c = i32x4::new(921070544, 1408695249, -136396947, -1525372302);
    let r = i64x2::new(-9168294401733980319, -6685995888074347700);

    assert_eq!(
        r,
        transmute(lsx_vmsub_w(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmsub_d() {
    let a = i64x2::new(-5022267712807149796, 8788062746333130381);
    let b = i64x2::new(594946727227821886, -4907188100068238790);
    let c = i64x2::new(-5753096081940451712, 2150588928473907718);
    let r = i64x2::new(-734195902542963684, -4942536302810424015);

    assert_eq!(
        r,
        transmute(lsx_vmsub_d(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vdiv_b() {
    let a = i8x16::new(
        56, 78, 12, -67, -45, -79, 3, -81, 85, 97, 41, -86, 106, -102, 35, 59,
    );
    let b = i8x16::new(
        48, -92, -93, -74, -32, 113, 86, -8, -99, -21, -14, -19, 124, -113, 29, -120,
    );
    let r = i64x2::new(720575944674246657, 281475060530176);

    assert_eq!(r, transmute(lsx_vdiv_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vdiv_h() {
    let a = i16x8::new(17409, -1878, -20289, -20815, 23275, 32438, 27688, 29943);
    let b = i16x8::new(-11221, 24673, 19931, 3799, -3251, -21373, -13758, -31286);
    let r = i64x2::new(-1125904201744385, 281470681743353);

    assert_eq!(r, transmute(lsx_vdiv_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vdiv_w() {
    let a = i32x4::new(912619458, 297234237, 1790081728, 1556369143);
    let b = i32x4::new(-775731190, 1887886939, 1001718213, 1135075421);
    let r = i64x2::new(4294967295, 4294967297);

    assert_eq!(r, transmute(lsx_vdiv_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vdiv_d() {
    let a = i64x2::new(8060378764891126625, 720122833079320324);
    let b = i64x2::new(-9175012156877545557, -6390704898809702209);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vdiv_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vdiv_bu() {
    let a = u8x16::new(
        153, 216, 32, 99, 9, 152, 44, 162, 131, 155, 164, 32, 248, 152, 88, 220,
    );
    let b = u8x16::new(
        27, 125, 253, 245, 104, 196, 141, 201, 107, 65, 51, 126, 107, 90, 130, 185,
    );
    let r = i64x2::new(261, 72058702139687425);

    assert_eq!(r, transmute(lsx_vdiv_bu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vdiv_hu() {
    let a = u16x8::new(47825, 17349, 21777, 60576, 31104, 31380, 8974, 51905);
    let b = u16x8::new(25282, 44917, 13706, 63351, 58837, 46710, 29092, 57823);
    let r = i64x2::new(4294967297, 0);

    assert_eq!(r, transmute(lsx_vdiv_hu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vdiv_wu() {
    let a = u32x4::new(1861719625, 952645030, 2402876315, 3695614684);
    let b = u32x4::new(1130189258, 1211056894, 2357258312, 3855913706);
    let r = i64x2::new(1, 1);

    assert_eq!(r, transmute(lsx_vdiv_wu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vdiv_du() {
    let a = u64x2::new(7958239212167095743, 5349587769754015194);
    let b = u64x2::new(14945948123666054968, 10864054932328247404);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vdiv_du(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vhaddw_h_b() {
    let a = i8x16::new(
        33, -91, 3, -119, 28, -34, -19, -51, 41, -83, 102, 116, 45, 50, -94, 121,
    );
    let b = i8x16::new(
        49, 50, 108, -49, -44, -25, 99, 7, -101, 39, -125, 11, -21, -99, -123, 29,
    );
    let r = i64x2::new(13791943145684950, -562821104926904);

    assert_eq!(r, transmute(lsx_vhaddw_h_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vhaddw_w_h() {
    let a = i16x8::new(-20323, -26647, 21748, 24233, 27893, -27604, 16391, 14873);
    let b = i16x8::new(
        -10851, -15249, -11124, -22012, -32205, -17044, 27739, -19038,
    );
    let r = i64x2::new(56307021213062, 183021441324639);

    assert_eq!(r, transmute(lsx_vhaddw_w_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vhaddw_d_w() {
    let a = i32x4::new(1127296124, -1382562520, -1791538949, 534516309);
    let b = i32x4::new(-1119468785, -1334232049, -1752131604, -2016112631);
    let r = i64x2::new(-2502031305, -1217615295);

    assert_eq!(r, transmute(lsx_vhaddw_d_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vhaddw_hu_bu() {
    let a = u8x16::new(
        72, 148, 45, 246, 151, 252, 69, 31, 91, 247, 215, 57, 125, 49, 141, 27,
    );
    let b = u8x16::new(
        76, 120, 158, 172, 253, 12, 131, 16, 18, 131, 114, 207, 1, 100, 48, 141,
    );
    let r = i64x2::new(45601115212087520, 21110838012870921);

    assert_eq!(r, transmute(lsx_vhaddw_hu_bu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vhaddw_wu_hu() {
    let a = u16x8::new(46665, 29041, 34462, 31370, 18289, 12579, 33777, 52188);
    let b = u16x8::new(40369, 53005, 64424, 35720, 9231, 19965, 20662, 8208);
    let r = i64x2::new(411432097222434, 312888367535410);

    assert_eq!(r, transmute(lsx_vhaddw_wu_hu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vhaddw_du_wu() {
    let a = u32x4::new(3058953381, 3443284865, 3364703869, 2180288462);
    let b = u32x4::new(728838120, 1267673009, 2659634151, 2264611356);
    let r = i64x2::new(4172122985, 4839922613);

    assert_eq!(r, transmute(lsx_vhaddw_du_wu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vhsubw_h_b() {
    let a = i8x16::new(
        20, -94, 56, 36, -78, -53, -65, 62, -23, 3, -26, 16, -36, 92, -87, -21,
    );
    let b = i8x16::new(
        -45, -92, 19, 45, -108, 44, 78, -127, -49, 23, -6, -3, 24, -8, 90, 51,
    );
    let r = i64x2::new(-4503363402989617, -31243430355664844);

    assert_eq!(r, transmute(lsx_vhsubw_h_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vhsubw_w_h() {
    let a = i16x8::new(-32636, -15640, 17489, 24551, 28768, 8187, -7376, -16756);
    let b = i16x8::new(-14204, -13312, 8240, -4455, -6362, -4711, -30790, -15773);
    let r = i64x2::new(70059506530916, 60275571046613);

    assert_eq!(r, transmute(lsx_vhsubw_w_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vhsubw_d_w() {
    let a = i32x4::new(-1518455529, -1873161613, -1441786902, 713965134);
    let b = i32x4::new(-1671723008, 870456702, 264823818, 13322401);
    let r = i64x2::new(-201438605, 449141316);

    assert_eq!(r, transmute(lsx_vhsubw_d_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vhsubw_hu_bu() {
    let a = u8x16::new(
        67, 78, 163, 156, 17, 58, 245, 19, 180, 161, 166, 207, 240, 5, 221, 157,
    );
    let b = u8x16::new(
        122, 131, 70, 56, 162, 5, 241, 241, 43, 5, 7, 236, 195, 26, 6, 17,
    );
    let r = i64x2::new(-62206416523952172, 42783380429340790);

    assert_eq!(r, transmute(lsx_vhsubw_hu_bu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vhsubw_wu_hu() {
    let a = u16x8::new(48161, 61606, 48243, 42252, 5643, 40672, 13711, 1172);
    let b = u16x8::new(5212, 32159, 36502, 59290, 7604, 229, 35511, 47443);
    let r = i64x2::new(24696062008394, -147484881944276);

    assert_eq!(r, transmute(lsx_vhsubw_wu_hu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vhsubw_du_wu() {
    let a = u32x4::new(2721083043, 781151638, 4268150742, 392308867);
    let b = u32x4::new(1383087137, 2403951939, 360532131, 3513614550);
    let r = i64x2::new(-601935499, 31776736);

    assert_eq!(r, transmute(lsx_vhsubw_du_wu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmod_b() {
    let a = i8x16::new(
        -89, -117, 89, -114, -65, 67, -20, 38, -38, -118, 30, 91, -16, -100, -109, -35,
    );
    let b = i8x16::new(
        94, -92, -13, 26, -6, -121, 39, -114, 74, -108, 95, 108, -65, -21, 67, 92,
    );
    let r = i64x2::new(2804691417388804007, -2461515231199824166);

    assert_eq!(r, transmute(lsx_vmod_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmod_h() {
    let a = i16x8::new(-29453, 12108, 10947, 28516, 4854, 1994, -30042, -18472);
    let b = i16x8::new(1550, 9221, -12080, 14553, -24847, 28286, 1074, 192);
    let r = i64x2::new(3930282117007147005, -10982007906888970);

    assert_eq!(r, transmute(lsx_vmod_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmod_w() {
    let a = i32x4::new(-2061299866, -1170666395, -1617297141, 594549537);
    let b = i32x4::new(344507881, 1692387020, -1397506903, -1257953510);
    let r = i64x2::new(-5027973877095011085, 2553570821342119010);

    assert_eq!(r, transmute(lsx_vmod_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmod_d() {
    let a = i64x2::new(-6018318621764124581, -5715738494441059378);
    let b = i64x2::new(4636642606889723746, -259899475747531088);
    let r = i64x2::new(-1381676014874400835, -257849503742906530);

    assert_eq!(r, transmute(lsx_vmod_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmod_bu() {
    let a = u8x16::new(
        122, 163, 72, 171, 64, 10, 201, 101, 196, 162, 190, 86, 253, 173, 221, 65,
    );
    let b = u8x16::new(
        186, 243, 157, 205, 48, 190, 55, 245, 72, 203, 140, 64, 8, 25, 252, 227,
    );
    let r = i64x2::new(7287961163701724026, 4745974892933063220);

    assert_eq!(r, transmute(lsx_vmod_bu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmod_hu() {
    let a = u16x8::new(26509, 32785, 35218, 8560, 18289, 13375, 35585, 60973);
    let b = u16x8::new(15317, 24954, 61354, 3720, 21471, 6193, 8193, 35745);
    let r = i64x2::new(315403234587388856, 7101062794264266609);

    assert_eq!(r, transmute(lsx_vmod_hu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmod_wu() {
    let a = u32x4::new(3940871454, 2498938081, 2241198148, 777660345);
    let b = u32x4::new(49228057, 2249712923, 358897384, 1782599598);
    let r = i64x2::new(1070413902953059662, 3340025749258890964);

    assert_eq!(r, transmute(lsx_vmod_wu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmod_du() {
    let a = u64x2::new(7747010922784437137, 16089799939101946183);
    let b = u64x2::new(16850073055169051895, 16069565262862467484);
    let r = i64x2::new(7747010922784437137, 20234676239478699);

    assert_eq!(r, transmute(lsx_vmod_du(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vreplve_b() {
    let a = i8x16::new(
        -62, -110, -89, -84, -11, -37, 90, -28, -41, -37, -53, 123, -55, 22, 20, -80,
    );
    let r = i64x2::new(-2893606913523066921, -2893606913523066921);

    assert_eq!(r, transmute(lsx_vreplve_b(transmute(a), -8)));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vreplve_h() {
    let a = i16x8::new(-29429, -23495, 8705, -7614, -25353, 11887, -25989, -12818);
    let r = i64x2::new(-3607719825936298514, -3607719825936298514);

    assert_eq!(r, transmute(lsx_vreplve_h(transmute(a), 7)));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vreplve_w() {
    let a = i32x4::new(1584940676, 95787593, -1655264847, 682404402);
    let r = i64x2::new(411404579393346121, 411404579393346121);

    assert_eq!(r, transmute(lsx_vreplve_w(transmute(a), -3)));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vreplve_d() {
    let a = i64x2::new(7614424214598615675, -7096892795239148002);
    let r = i64x2::new(7614424214598615675, 7614424214598615675);

    assert_eq!(r, transmute(lsx_vreplve_d(transmute(a), 0)));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vreplvei_b() {
    let a = i8x16::new(
        62, -120, 10, 58, 124, -30, 57, -78, -114, 6, -39, 46, 58, -72, -44, 21,
    );
    let r = i64x2::new(-2097865012304223518, -2097865012304223518);

    assert_eq!(r, transmute(lsx_vreplvei_b::<5>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vreplvei_h() {
    let a = i16x8::new(-15455, -4410, 5029, 25863, -23170, 26570, 27423, -834);
    let r = i64x2::new(7719006069021698847, 7719006069021698847);

    assert_eq!(r, transmute(lsx_vreplvei_h::<6>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vreplvei_w() {
    let a = i32x4::new(1843143434, 491125746, -328585251, -1996512058);
    let r = i64x2::new(7916240772710277898, 7916240772710277898);

    assert_eq!(r, transmute(lsx_vreplvei_w::<0>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vreplvei_d() {
    let a = i64x2::new(4333963848299154309, -8310246545782080694);
    let r = i64x2::new(-8310246545782080694, -8310246545782080694);

    assert_eq!(r, transmute(lsx_vreplvei_d::<1>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpickev_b() {
    let a = i8x16::new(
        89, 84, -94, 3, 41, -86, -10, 120, 62, -102, 44, -88, 12, -75, -13, 65,
    );
    let b = i8x16::new(
        -31, 44, -76, -76, 52, -71, 44, -110, -4, 124, -38, 76, 108, 43, 54, 60,
    );
    let r = i64x2::new(3921750152141124833, -933322373843017127);

    assert_eq!(r, transmute(lsx_vpickev_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpickev_h() {
    let a = i16x8::new(-5994, -14344, -28338, -25788, 5710, 1638, 494, -2554);
    let b = i16x8::new(-5248, -1786, -21768, 23214, -4223, 23538, -24936, -32316);
    let r = i64x2::new(-7018596679058658432, 139073165196191894);

    assert_eq!(r, transmute(lsx_vpickev_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpickev_w() {
    let a = i32x4::new(548489620, -968269400, -179106837, -1739507044);
    let b = i32x4::new(-1187277846, -787064901, -980229113, 1746235326);
    let r = i64x2::new(-4210051979814398998, -769258006856513132);

    assert_eq!(r, transmute(lsx_vpickev_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpickev_d() {
    let a = i64x2::new(1789073368466131160, 9168587701455881156);
    let b = i64x2::new(6574352346370076190, -3979792156310826694);
    let r = i64x2::new(6574352346370076190, 1789073368466131160);

    assert_eq!(r, transmute(lsx_vpickev_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpickod_b() {
    let a = i8x16::new(
        -125, 4, -27, 25, 117, 98, -51, -93, -37, 110, -127, 115, 114, -108, 74, -85,
    );
    let b = i8x16::new(
        93, -72, 89, 104, 84, 15, 77, 74, 91, -34, 118, -108, 13, 21, 105, 114,
    );
    let r = i64x2::new(8220640377280882872, -6083110277645985532);

    assert_eq!(r, transmute(lsx_vpickod_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpickod_h() {
    let a = i16x8::new(1454, -18740, 13146, 10497, 4897, 31962, 19208, 21910);
    let b = i16x8::new(12047, 25024, -10709, -28077, 24357, 19934, 10289, 28546);
    let r = i64x2::new(8035070303515402688, 6167254016163165900);

    assert_eq!(r, transmute(lsx_vpickod_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpickod_w() {
    let a = i32x4::new(869069429, -1916930406, 1864611728, -1640302268);
    let b = i32x4::new(-99240403, 314407358, 543396756, 1976776696);
    let r = i64x2::new(8490191261129341374, -7045044594236590438);

    assert_eq!(r, transmute(lsx_vpickod_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpickod_d() {
    let a = i64x2::new(7031942541839550339, -7578696032343374601);
    let b = i64x2::new(-4197243771252175958, -543692393753629390);
    let r = i64x2::new(-543692393753629390, -7578696032343374601);

    assert_eq!(r, transmute(lsx_vpickod_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vilvh_b() {
    let a = i8x16::new(
        -58, -103, -5, 33, 124, -24, -18, 20, 22, -100, -6, 16, 40, 89, -41, -37,
    );
    let b = i8x16::new(
        -42, 76, 46, -4, 67, 45, 99, -7, 63, 20, 113, -50, 67, -23, -20, 112,
    );
    let r = i64x2::new(1211180715666052671, -2634368371891034045);

    assert_eq!(r, transmute(lsx_vilvh_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vilvh_h() {
    let a = i16x8::new(24338, 259, -22693, 16519, -28272, -16751, 1883, 16217);
    let b = i16x8::new(23768, -31845, 28689, 14757, 9499, 7795, -13573, -10011);
    let r = i64x2::new(-4714953853167983333, 4564918175499275003);

    assert_eq!(r, transmute(lsx_vilvh_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vilvh_w() {
    let a = i32x4::new(-968342074, -1976160649, -1249304918, -279518364);
    let b = i32x4::new(-737076987, 38515006, 602108871, -63099569);
    let r = i64x2::new(-5365723764939852857, -1200522227779556017);

    assert_eq!(r, transmute(lsx_vilvh_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vilvh_d() {
    let a = i64x2::new(2505149669372896333, 5375050218784453679);
    let b = i64x2::new(-2160658667838026389, 1449429407527660400);
    let r = i64x2::new(1449429407527660400, 5375050218784453679);

    assert_eq!(r, transmute(lsx_vilvh_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vilvl_b() {
    let a = i8x16::new(
        57, 109, 61, 96, 101, 69, -42, 118, 112, -17, 63, 68, -54, 32, 17, -122,
    );
    let b = i8x16::new(
        -48, -30, -102, 100, -3, 85, 100, 46, 82, 67, -20, -56, 93, 96, -39, 108,
    );
    let r = i64x2::new(6945744258789947856, 8515979671552484861);

    assert_eq!(r, transmute(lsx_vilvl_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vilvl_h() {
    let a = i16x8::new(28844, -23308, 4163, -8033, 12472, -16423, 14534, 31242);
    let b = i16x8::new(11601, 6788, 3174, -4208, -25999, -25660, -4591, 7133);
    let r = i64x2::new(-6560589601043632815, -2260825085889541018);

    assert_eq!(r, transmute(lsx_vilvl_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vilvl_w() {
    let a = i32x4::new(-997094955, 1731171907, 1528236839, -646874689);
    let b = i32x4::new(486029703, 1245981961, 112180197, 1939621508);
    let r = i64x2::new(-4282490222245561977, 7435326725564935433);

    assert_eq!(r, transmute(lsx_vilvl_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vilvl_d() {
    let a = i64x2::new(7063413230460842607, -4234618008113981723);
    let b = i64x2::new(3142531875873363679, 736682102982019415);
    let r = i64x2::new(3142531875873363679, 7063413230460842607);

    assert_eq!(r, transmute(lsx_vilvl_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpackev_b() {
    let a = i8x16::new(
        63, 38, -47, 98, 19, 68, -27, 1, 108, 65, 108, 31, -102, 37, -27, 50,
    );
    let b = i8x16::new(
        59, 11, -44, 73, -74, -15, 61, 17, -37, 117, -39, 28, 38, 49, -34, -86,
    );
    let r = i64x2::new(-1928363389519380677, -1882898104368665381);

    assert_eq!(r, transmute(lsx_vpackev_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpackev_h() {
    let a = i16x8::new(26574, -30949, 26762, -28439, 5382, -25386, 5192, -9816);
    let b = i16x8::new(-9444, 5210, -14402, 17972, 16606, 2450, 5123, 14727);
    let r = i64x2::new(7533052947329899292, 1461440082551914718);

    assert_eq!(r, transmute(lsx_vpackev_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpackev_w() {
    let a = i32x4::new(1312465803, -1752635324, -1943199176, -362848304);
    let b = i32x4::new(-872903277, 1255047449, -2110158279, 682925573);
    let r = i64x2::new(5636997704425442707, -8345976908349339079);

    assert_eq!(r, transmute(lsx_vpackev_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpackev_d() {
    let a = i64x2::new(7118943335298607169, 3038173153862744209);
    let b = i64x2::new(-9119315954224042738, -4563700463464702181);
    let r = i64x2::new(-9119315954224042738, 7118943335298607169);

    assert_eq!(r, transmute(lsx_vpackev_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpackod_b() {
    let a = i8x16::new(
        94, -48, 43, -58, -47, 27, -33, 60, 50, -38, 41, -41, 76, -46, 103, -60,
    );
    let b = i8x16::new(
        -117, -11, 72, -9, -99, -52, -102, -22, -7, -8, 8, -65, 101, 29, 86, 27,
    );
    let r = i64x2::new(4389351353151377653, -4315624792288929032);

    assert_eq!(r, transmute(lsx_vpackod_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpackod_h() {
    let a = i16x8::new(-18827, 19151, 4246, -15752, -1028, 29166, 3421, -32610);
    let b = i16x8::new(-23247, 17928, -13353, -20146, 5696, 22071, -10728, -30262);
    let r = i64x2::new(-4433598883325590008, -9178747487946648009);

    assert_eq!(r, transmute(lsx_vpackod_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpackod_w() {
    let a = i32x4::new(-1183976810, 11929980, -1445863799, 1567314918);
    let b = i32x4::new(445270781, 793617340, -1461557030, -22199234);
    let r = i64x2::new(51238874735551420, 6731566319615689790);

    assert_eq!(r, transmute(lsx_vpackod_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpackod_d() {
    let a = i64x2::new(-4549504442184266063, -4670773907187480618);
    let b = i64x2::new(9039771682296134623, -6404442538060227683);
    let r = i64x2::new(-6404442538060227683, -4670773907187480618);

    assert_eq!(r, transmute(lsx_vpackod_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vshuf_h() {
    let a = i16x8::new(7, 12, 6, 8, 11, 2, 4, 7);
    let b = i16x8::new(19221, 5841, 2738, -31394, -31337, -27662, 24655, 28090);
    let c = i16x8::new(27835, 20061, 7214, -10489, -14005, -27870, -12303, 14443);
    let r = i64x2::new(5410459163590867051, 4065564413064545630);

    assert_eq!(
        r,
        transmute(lsx_vshuf_h(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vshuf_w() {
    let a = i32x4::new(0, 3, 4, 6);
    let b = i32x4::new(921730307, -1175025178, 241337062, 53139449);
    let c = i32x4::new(-67250654, 55397321, 1170999941, 1704507894);
    let r = i64x2::new(7320805664731551266, 1036534789524454659);

    assert_eq!(
        r,
        transmute(lsx_vshuf_w(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vshuf_d() {
    let a = i64x2::new(1, 2);
    let b = i64x2::new(4033696695079994582, -3146912063343863773);
    let c = i64x2::new(-4786751363389755273, 1769232540309840996);
    let r = i64x2::new(1769232540309840996, 4033696695079994582);

    assert_eq!(
        r,
        transmute(lsx_vshuf_d(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vand_v() {
    let a = u8x16::new(
        105, 106, 193, 101, 82, 63, 227, 23, 246, 17, 117, 134, 98, 233, 41, 128,
    );
    let b = u8x16::new(
        254, 161, 164, 46, 166, 61, 123, 67, 90, 217, 49, 98, 166, 236, 128, 175,
    );
    let r = i64x2::new(244105884219744360, -9223116804091473582);

    assert_eq!(r, transmute(lsx_vand_v(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vandi_b() {
    let a = u8x16::new(
        167, 0, 108, 41, 255, 45, 24, 175, 229, 222, 89, 15, 63, 15, 187, 213,
    );
    let r = i64x2::new(-8135737750142058361, -7666517314596397435);

    assert_eq!(r, transmute(lsx_vandi_b::<159>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vor_v() {
    let a = u8x16::new(
        87, 193, 209, 232, 106, 36, 72, 199, 202, 213, 174, 2, 78, 181, 135, 178,
    );
    let b = u8x16::new(
        253, 19, 178, 143, 132, 123, 29, 28, 200, 36, 9, 212, 12, 35, 164, 169,
    );
    let r = i64x2::new(-2351582766212852737, -4924766118269159990);

    assert_eq!(r, transmute(lsx_vor_v(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vori_b() {
    let a = u8x16::new(
        134, 61, 120, 206, 181, 179, 192, 181, 115, 179, 137, 110, 147, 51, 93, 65,
    );
    let r = i64x2::new(-589140355308650538, -3179554720060804109);

    assert_eq!(r, transmute(lsx_vori_b::<210>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vnor_v() {
    let a = u8x16::new(
        116, 165, 106, 148, 116, 117, 91, 213, 195, 131, 160, 33, 223, 207, 12, 147,
    );
    let b = u8x16::new(
        242, 233, 135, 143, 129, 199, 130, 192, 222, 143, 223, 103, 232, 53, 98, 129,
    );
    let r = i64x2::new(3036560889408918025, 7823034030269427744);

    assert_eq!(r, transmute(lsx_vnor_v(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vnori_b() {
    let a = u8x16::new(
        142, 138, 177, 202, 121, 170, 99, 149, 251, 153, 234, 191, 10, 185, 182, 212,
    );
    let r = i64x2::new(5227628601268782144, 596802560304890884);

    assert_eq!(r, transmute(lsx_vnori_b::<51>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vxor_v() {
    let a = u8x16::new(
        33, 58, 188, 69, 128, 23, 145, 174, 229, 254, 21, 227, 196, 131, 115, 100,
    );
    let b = u8x16::new(
        10, 61, 91, 105, 232, 114, 191, 215, 83, 11, 124, 157, 132, 242, 94, 59,
    );
    let r = i64x2::new(8732028225622312747, 6858262329367852470);

    assert_eq!(r, transmute(lsx_vxor_v(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vxori_b() {
    let a = u8x16::new(
        27, 105, 197, 119, 145, 141, 167, 209, 51, 206, 89, 42, 45, 215, 239, 160,
    );
    let r = i64x2::new(3478586993001400570, 4687744515358339026);

    assert_eq!(r, transmute(lsx_vxori_b::<225>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitsel_v() {
    let a = u8x16::new(
        217, 159, 221, 209, 154, 9, 59, 230, 33, 109, 205, 229, 188, 222, 1, 94,
    );
    let b = u8x16::new(
        49, 116, 245, 6, 184, 146, 9, 1, 133, 27, 12, 4, 47, 11, 8, 133,
    );
    let c = u8x16::new(
        140, 105, 10, 4, 218, 82, 128, 160, 67, 218, 139, 14, 248, 53, 35, 81,
    );
    let r = i64x2::new(5060668949517432401, 1081087304254897953);

    assert_eq!(
        r,
        transmute(lsx_vbitsel_v(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbitseli_b() {
    let a = u8x16::new(
        224, 93, 78, 91, 41, 115, 130, 96, 34, 22, 227, 254, 0, 44, 237, 193,
    );
    let b = u8x16::new(
        138, 4, 83, 190, 229, 199, 235, 99, 62, 236, 201, 78, 160, 181, 45, 187,
    );
    let r = i64x2::new(4857631126842327370, 8881540057610709020);

    assert_eq!(
        r,
        transmute(lsx_vbitseli_b::<65>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vshuf4i_b() {
    let a = i8x16::new(
        -83, 65, -54, 44, -52, -97, -93, 54, 118, -10, -20, -43, -60, -86, -116, -47,
    );
    let r = i64x2::new(3937170420478429898, -3347145886530736916);

    assert_eq!(r, transmute(lsx_vshuf4i_b::<234>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vshuf4i_h() {
    let a = i16x8::new(27707, -1094, -15784, -28387, 31634, -12323, -30387, -11480);
    let r = i64x2::new(-7989953385787032646, -3231104182470389795);

    assert_eq!(r, transmute(lsx_vshuf4i_h::<209>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vshuf4i_w() {
    let a = i32x4::new(768986805, -1036149600, -1196682940, -214444511);
    let r = i64x2::new(3302773179299516085, -5139714087882845884);

    assert_eq!(r, transmute(lsx_vshuf4i_w::<160>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vreplgr2vr_b() {
    let r = i64x2::new(795741901218843403, 795741901218843403);

    assert_eq!(r, transmute(lsx_vreplgr2vr_b(970839819)));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vreplgr2vr_h() {
    let r = i64x2::new(-6504141532176800324, -6504141532176800324);

    assert_eq!(r, transmute(lsx_vreplgr2vr_h(93693372)));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vreplgr2vr_w() {
    let r = i64x2::new(-6737078705572473188, -6737078705572473188);

    assert_eq!(r, transmute(lsx_vreplgr2vr_w(-1568598372)));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vreplgr2vr_d() {
    let r = i64x2::new(5000134708087557572, 5000134708087557572);

    assert_eq!(r, transmute(lsx_vreplgr2vr_d(5000134708087557572)));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpcnt_b() {
    let a = i8x16::new(
        29, -96, 22, 17, 38, -51, -97, 82, 17, -82, -30, -42, -44, 107, -51, 80,
    );
    let r = i64x2::new(217867142450840068, 145528077781566722);

    assert_eq!(r, transmute(lsx_vpcnt_b(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpcnt_h() {
    let a = i16x8::new(-512, 10388, -21267, -27094, 1085, -26444, -29360, -11576);
    let r = i64x2::new(1970367786975239, 1970350607237126);

    assert_eq!(r, transmute(lsx_vpcnt_h(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpcnt_w() {
    let a = i32x4::new(1399276601, -2094725994, -100739325, -1239551533);
    let r = i64x2::new(47244640271, 81604378645);

    assert_eq!(r, transmute(lsx_vpcnt_w(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpcnt_d() {
    let a = i64x2::new(-4470823169399930539, 3184270543884128372);
    let r = i64x2::new(29, 25);

    assert_eq!(r, transmute(lsx_vpcnt_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vclo_b() {
    let a = i8x16::new(
        94, 66, -88, -43, 113, 10, 5, -96, 96, 78, 3, -30, -24, -29, 20, 115,
    );
    let r = i64x2::new(72057594071547904, 3311470116864);

    assert_eq!(r, transmute(lsx_vclo_b(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vclo_h() {
    let a = i16x8::new(-5432, 27872, -9150, 27393, 25236, 1028, -21312, -25189);
    let r = i64x2::new(8589934595, 281479271677952);

    assert_eq!(r, transmute(lsx_vclo_h(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vclo_w() {
    let a = i32x4::new(1214322611, -1755838761, -1222326743, -1511364419);
    let r = i64x2::new(4294967296, 4294967297);

    assert_eq!(r, transmute(lsx_vclo_w(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vclo_d() {
    let a = i64x2::new(-249299854527467825, -459308653408461862);
    let r = i64x2::new(6, 5);

    assert_eq!(r, transmute(lsx_vclo_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vclz_b() {
    let a = i8x16::new(
        -103, -39, -51, -74, -68, 126, -124, 33, 30, 54, -46, -53, -9, 96, 17, 74,
    );
    let r = i64x2::new(144116287587483648, 72903118479688195);

    assert_eq!(r, transmute(lsx_vclz_b(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vclz_h() {
    let a = i16x8::new(1222, 32426, 3164, -10763, 10189, -4197, -21841, -28676);
    let r = i64x2::new(17179934725, 2);

    assert_eq!(r, transmute(lsx_vclz_h(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vclz_w() {
    let a = i32x4::new(-490443689, -1039971379, -217310592, -1921086575);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vclz_w(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vclz_d() {
    let a = i64x2::new(4630351532137644314, -6587611980764816064);
    let r = i64x2::new(1, 0);

    assert_eq!(r, transmute(lsx_vclz_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpickve2gr_b() {
    let a = i8x16::new(
        119, 126, -107, -59, 22, -27, -67, 39, -66, -101, 34, -26, -16, 61, 20, 51,
    );
    let r: i32 = 51;

    assert_eq!(r, transmute(lsx_vpickve2gr_b::<15>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpickve2gr_h() {
    let a = i16x8::new(-12924, 31013, 18171, 20404, 21226, 14128, -6255, 26521);
    let r: i32 = 21226;

    assert_eq!(r, transmute(lsx_vpickve2gr_h::<4>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpickve2gr_w() {
    let a = i32x4::new(-1559379275, 2065542381, -1882161334, 1502157419);
    let r: i32 = -1882161334;

    assert_eq!(r, transmute(lsx_vpickve2gr_w::<2>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpickve2gr_d() {
    let a = i64x2::new(-6941380853339482104, 8405634758774935528);
    let r: i64 = -6941380853339482104;

    assert_eq!(r, transmute(lsx_vpickve2gr_d::<0>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpickve2gr_bu() {
    let a = i8x16::new(
        18, -111, 100, 2, -105, 20, 92, -40, -57, 117, 6, -119, -94, 86, -52, 35,
    );
    let r: u32 = 199;

    assert_eq!(r, transmute(lsx_vpickve2gr_bu::<8>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpickve2gr_hu() {
    let a = i16x8::new(25003, 5139, -12977, 7550, -12177, 19294, -2216, 12693);
    let r: u32 = 25003;

    assert_eq!(r, transmute(lsx_vpickve2gr_hu::<0>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpickve2gr_wu() {
    let a = i32x4::new(-295894883, 551663550, -710853968, 82692774);
    let r: u32 = 3999072413;

    assert_eq!(r, transmute(lsx_vpickve2gr_wu::<0>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpickve2gr_du() {
    let a = i64x2::new(748282319555413922, -1352335765832355666);
    let r: u64 = 748282319555413922;

    assert_eq!(r, transmute(lsx_vpickve2gr_du::<0>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vinsgr2vr_b() {
    let a = i8x16::new(
        58, 12, -107, 35, 111, -15, -99, 117, 119, 92, -18, 32, -44, -34, 53, -34,
    );
    let r = i64x2::new(8475195533421775930, -2423536021788533641);

    assert_eq!(
        r,
        transmute(lsx_vinsgr2vr_b::<14>(transmute(a), 1333652061))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vinsgr2vr_h() {
    let a = i16x8::new(-20591, 7819, 25287, -11296, 4604, 28833, -1306, 6418);
    let r = i64x2::new(-3179432729573085295, 1806782266980897276);

    assert_eq!(r, transmute(lsx_vinsgr2vr_h::<5>(transmute(a), -987420193)));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vinsgr2vr_w() {
    let a = i32x4::new(1608179655, 886830932, -621638499, 2021214690);
    let r = i64x2::new(3808909851629379527, 8681050995079237782);

    assert_eq!(r, transmute(lsx_vinsgr2vr_w::<2>(transmute(a), -960507754)));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vinsgr2vr_d() {
    let a = i64x2::new(-6562091001143116290, -2425423285843953307);
    let r = i64x2::new(-6562091001143116290, -233659266);

    assert_eq!(r, transmute(lsx_vinsgr2vr_d::<1>(transmute(a), -233659266)));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfadd_s() {
    let a = u32x4::new(1063501234, 1064367472, 1065334422, 1012846272);
    let b = u32x4::new(1050272808, 1054022924, 1064036136, 1063113730);
    let r = i64x2::new(4588396142719948771, 4567018621615066847);

    assert_eq!(r, transmute(lsx_vfadd_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfadd_d() {
    let a = u64x2::new(4602410992567934854, 4605792798803129629);
    let b = u64x2::new(4605819027271079334, 4601207158507578498);
    let r = i64x2::new(4608685566198055604, 4608371493448991663);

    assert_eq!(r, transmute(lsx_vfadd_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfsub_s() {
    let a = u32x4::new(1064451273, 1059693825, 1036187576, 1050580506);
    let b = u32x4::new(1063475462, 1045836432, 1065150677, 1042376676);
    let r = i64x2::new(4532926601401089072, 4475386505810184670);

    assert_eq!(r, transmute(lsx_vfsub_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfsub_d() {
    let a = u64x2::new(4601910797424251354, 4606993182294978423);
    let b = u64x2::new(4605973926398825814, 4600156145303017004);
    let r = i64x2::new(-4622342180736116526, 4603750919602422881);

    assert_eq!(r, transmute(lsx_vfsub_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfmul_s() {
    let a = u32x4::new(1060566900, 1061147127, 1010818944, 1053672244);
    let b = u32x4::new(1065241951, 1044285812, 1050678216, 1009264512);
    let r = i64x2::new(4471727895898079441, 4289440988347233543);

    assert_eq!(r, transmute(lsx_vfmul_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfmul_d() {
    let a = u64x2::new(4593483834506733144, 4602939512559809908);
    let b = u64x2::new(4605208047666947899, 4599634375243914522);
    let r = i64x2::new(4591550625791030606, 4595475933048682142);

    assert_eq!(r, transmute(lsx_vfmul_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfdiv_s() {
    let a = u32x4::new(1057501460, 1051070718, 1065221347, 1051828876);
    let b = u32x4::new(1055538538, 1042248668, 1061233585, 1063649172);
    let r = i64x2::new(4613180427594946541, 4523223175100126088);

    assert_eq!(r, transmute(lsx_vfdiv_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfdiv_d() {
    let a = u64x2::new(4591718910407182664, 4607068478646496456);
    let b = u64x2::new(4606326032528596062, 4601783079746725386);
    let r = i64x2::new(4592460108638699314, 4612120084672695832);

    assert_eq!(r, transmute(lsx_vfdiv_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcvt_h_s() {
    let a = u32x4::new(1020611712, 1046448896, 1062035346, 1052255382);
    let b = u32x4::new(1049501482, 1043939972, 1042291392, 1041250232);
    let r = i64x2::new(3495410141992989809, 3873441386606634666);

    assert_eq!(r, transmute(lsx_vfcvt_h_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcvt_s_d() {
    let a = u64x2::new(4586066291858051968, 4597324798333789044);
    let b = u64x2::new(4600251021237488420, 4593890179408150924);
    let r = i64x2::new(4469319308295208818, 4496796258465732597);

    assert_eq!(r, transmute(lsx_vfcvt_s_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfmin_s() {
    let a = u32x4::new(1016310272, 1064492378, 1043217948, 1060534856);
    let b = u32x4::new(1060093085, 1026130528, 1057322097, 1057646773);
    let r = i64x2::new(4407197060203522560, 4542558301798153756);

    assert_eq!(r, transmute(lsx_vfmin_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfmin_d() {
    let a = u64x2::new(4603437440563473519, 4603158282529654079);
    let b = u64x2::new(4584808359801648672, 4602712060570539582);
    let r = i64x2::new(4584808359801648672, 4602712060570539582);

    assert_eq!(r, transmute(lsx_vfmin_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfmina_s() {
    let a = u32x4::new(1061417856, 1052257408, 1056830440, 1055199170);
    let b = u32x4::new(1049119234, 1058336224, 1057046116, 1029386720);
    let r = i64x2::new(4519411155382848002, 4421182298393539560);

    assert_eq!(r, transmute(lsx_vfmina_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfmina_d() {
    let a = u64x2::new(4599160304044702024, 4603774209349450318);
    let b = u64x2::new(4599088744110071826, 4598732503789588496);
    let r = i64x2::new(4599088744110071826, 4598732503789588496);

    assert_eq!(r, transmute(lsx_vfmina_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfmax_s() {
    let a = u32x4::new(1054002242, 1061130492, 1034716288, 1064963760);
    let b = u32x4::new(1042175760, 1040826492, 1059132266, 1050815434);
    let r = i64x2::new(4557520760982391874, 4573984521684325226);

    assert_eq!(r, transmute(lsx_vfmax_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfmax_d() {
    let a = u64x2::new(4606275407710467505, 4593284088749839728);
    let b = u64x2::new(4593616624275112016, 4605244843740986156);
    let r = i64x2::new(4606275407710467505, 4605244843740986156);

    assert_eq!(r, transmute(lsx_vfmax_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfmaxa_s() {
    let a = u32x4::new(1059031357, 1043496676, 1044317464, 1055811838);
    let b = u32x4::new(1064739422, 1055122552, 1049654310, 1057411362);
    let r = i64x2::new(4531716855176798814, 4541547219258471462);

    assert_eq!(r, transmute(lsx_vfmaxa_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfmaxa_d() {
    let a = u64x2::new(4559235973242941440, 4606304546706191737);
    let b = u64x2::new(4603647289310579471, 4603999027307573908);
    let r = i64x2::new(4603647289310579471, 4606304546706191737);

    assert_eq!(r, transmute(lsx_vfmaxa_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfclass_s() {
    let a = u32x4::new(1059786314, 1058231666, 1061513647, 1038650488);
    let r = i64x2::new(549755814016, 549755814016);

    assert_eq!(r, transmute(lsx_vfclass_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfclass_d() {
    let a = u64x2::new(4601724705608768104, 4601126152607382566);
    let r = i64x2::new(128, 128);

    assert_eq!(r, transmute(lsx_vfclass_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfsqrt_s() {
    let a = u32x4::new(1055398716, 1050305974, 995168768, 1064901995);
    let r = i64x2::new(4543169501430832482, 4574681629207255333);

    assert_eq!(r, transmute(lsx_vfsqrt_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfsqrt_d() {
    let a = u64x2::new(4605784293613801157, 4602267946351406890);
    let r = i64x2::new(4606453893731357485, 4604397310232711799);

    assert_eq!(r, transmute(lsx_vfsqrt_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfrecip_s() {
    let a = u32x4::new(1003452672, 1050811504, 1044295808, 1064402913);
    let r = i64x2::new(4632552602764963931, 4577820515916044016);

    assert_eq!(r, transmute(lsx_vfrecip_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfrecip_d() {
    let a = u64x2::new(4598634931235673106, 4598630619264835010);
    let r = i64x2::new(4615355353482170689, 4615362460048142095);

    assert_eq!(r, transmute(lsx_vfrecip_d(transmute(a))));
}

#[simd_test(enable = "lsx,frecipe")]
unsafe fn test_lsx_vfrecipe_s() {
    let a = u32x4::new(1057583779, 1062308847, 1060089100, 1048454688);
    let r = i64x2::new(4583644530211711115, 4647978179615164140);

    assert_eq!(r, transmute(lsx_vfrecipe_s(transmute(a))));
}

#[simd_test(enable = "lsx,frecipe")]
unsafe fn test_lsx_vfrecipe_d() {
    let a = u64x2::new(4605515926442181274, 4605369703273365674);
    let r = i64x2::new(4608204937770303488, 4608317161507651584);

    assert_eq!(r, transmute(lsx_vfrecipe_d(transmute(a))));
}

#[simd_test(enable = "lsx,frecipe")]
unsafe fn test_lsx_vfrsqrte_s() {
    let a = u32x4::new(1064377488, 1055815904, 1056897740, 1064016656);
    let r = i64x2::new(4592421282989204764, 4577184195020153336);

    assert_eq!(r, transmute(lsx_vfrsqrte_s(transmute(a))));
}

#[simd_test(enable = "lsx,frecipe")]
unsafe fn test_lsx_vfrsqrte_d() {
    let a = u64x2::new(4602766865443628663, 4605323203937791867);
    let r = i64x2::new(4608986772678901760, 4607734355383549952);

    assert_eq!(r, transmute(lsx_vfrsqrte_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfrint_s() {
    let a = u32x4::new(1062138521, 1056849108, 1034089720, 1038314384);
    let r = i64x2::new(1065353216, 0);

    assert_eq!(r, transmute(lsx_vfrint_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfrint_d() {
    let a = u64x2::new(4598620052333442366, 4603262362368837514);
    let r = i64x2::new(0, 4607182418800017408);

    assert_eq!(r, transmute(lsx_vfrint_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfrsqrt_s() {
    let a = u32x4::new(1058614029, 1050504950, 1013814976, 1062355001);
    let r = i64x2::new(4604601921912011494, 4579384257679777264);

    assert_eq!(r, transmute(lsx_vfrsqrt_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfrsqrt_d() {
    let a = u64x2::new(4602924191185043139, 4606088351077917251);
    let r = i64x2::new(4608881149202581394, 4607483676176768181);

    assert_eq!(r, transmute(lsx_vfrsqrt_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vflogb_s() {
    let a = u32x4::new(1053488512, 1061429282, 1064965594, 1061326585);
    let r = i64x2::new(-4647714812225126400, -4647714812233515008);

    assert_eq!(r, transmute(lsx_vflogb_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vflogb_d() {
    let a = u64x2::new(4589481276789128632, 4599408395082246526);
    let r = i64x2::new(-4607182418800017408, -4611686018427387904);

    assert_eq!(r, transmute(lsx_vflogb_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcvth_s_h() {
    let a = i16x8::new(29550, -13884, 689, -1546, 24006, -19112, -12769, 1779);
    let r = i64x2::new(-4707668984349540352, 4097818267320836096);

    assert_eq!(r, transmute(lsx_vfcvth_s_h(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcvth_d_s() {
    let a = u32x4::new(1051543000, 1042275304, 1038283216, 1063876621);
    let r = i64x2::new(4592649323212177408, 4606389677895712768);

    assert_eq!(r, transmute(lsx_vfcvth_d_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcvtl_s_h() {
    let a = i16x8::new(-21951, -13772, -17190, 9566, -19227, 9682, 13427, -30861);
    let r = i64x2::new(-4519784435355738112, 4371798972740354048);

    assert_eq!(r, transmute(lsx_vfcvtl_s_h(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcvtl_d_s() {
    let a = u32x4::new(1059809930, 1051084496, 1062618346, 1058273673);
    let r = i64x2::new(4604206389789720576, 4599521958080544768);

    assert_eq!(r, transmute(lsx_vfcvtl_d_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftint_w_s() {
    let a = u32x4::new(1064738153, 1040181800, 1064331056, 1050732566);
    let r = i64x2::new(1, 1);

    assert_eq!(r, transmute(lsx_vftint_w_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftint_l_d() {
    let a = u64x2::new(4602244632405616462, 4606437548563176328);
    let r = i64x2::new(0, 1);

    assert_eq!(r, transmute(lsx_vftint_l_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftint_wu_s() {
    let a = u32x4::new(1051598962, 1051261298, 1059326008, 1057784192);
    let r = i64x2::new(0, 4294967297);

    assert_eq!(r, transmute(lsx_vftint_wu_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftint_lu_d() {
    let a = u64x2::new(4605561240422589260, 4595241299507769712);
    let r = i64x2::new(1, 0);

    assert_eq!(r, transmute(lsx_vftint_lu_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrz_w_s() {
    let a = u32x4::new(1027659872, 1064207676, 1058472873, 1055740014);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vftintrz_w_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrz_l_d() {
    let a = u64x2::new(4605051539601556532, 4605129242354661923);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vftintrz_l_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrz_wu_s() {
    let a = u32x4::new(1060876751, 1053710034, 1057340881, 1055555596);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vftintrz_wu_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrz_lu_d() {
    let a = u64x2::new(4598711097624940956, 4598268778109474002);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vftintrz_lu_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vffint_s_w() {
    let a = i32x4::new(81337967, 1396520141, 2124859806, 1655115736);
    let r = i64x2::new(5667351778062705614, 5676028806041521555);

    assert_eq!(r, transmute(lsx_vffint_s_w(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vffint_d_l() {
    let a = i64x2::new(-1543454772280682525, -7672333112582708041);
    let r = i64x2::new(-4344448119835677720, -4333977527979901593);

    assert_eq!(r, transmute(lsx_vffint_d_l(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vffint_s_wu() {
    let a = u32x4::new(2224947834, 194720725, 2248289069, 1131100007);
    let r = i64x2::new(5564675890493038082, 5658445755393114667);

    assert_eq!(r, transmute(lsx_vffint_s_wu(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vffint_d_lu() {
    let a = u64x2::new(11793247389644223387, 1356636411353166515);
    let r = i64x2::new(4892164017273962878, 4878194157796724979);

    assert_eq!(r, transmute(lsx_vffint_d_lu(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vandn_v() {
    let a = u8x16::new(
        69, 83, 176, 218, 73, 205, 105, 229, 131, 233, 158, 58, 63, 68, 94, 223,
    );
    let b = u8x16::new(
        12, 197, 21, 164, 196, 200, 144, 3, 232, 91, 46, 182, 156, 14, 53, 106,
    );
    let r = i64x2::new(184648152262214664, 2315143230533931624);

    assert_eq!(r, transmute(lsx_vandn_v(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vneg_b() {
    let a = i8x16::new(
        -118, -51, 32, 96, -18, 11, -3, 86, 77, 78, -120, 105, -47, 6, -127, -49,
    );
    let r = i64x2::new(-6195839201974406282, 3566844512212398771);

    assert_eq!(r, transmute(lsx_vneg_b(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vneg_h() {
    let a = i16x8::new(-6540, 25893, -2534, 29805, -28719, -16331, -20168, 14650);
    let r = i64x2::new(-8389350794815923828, -4123521786840387537);

    assert_eq!(r, transmute(lsx_vneg_h(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vneg_w() {
    let a = i32x4::new(-927815384, -898911982, 716171852, -2025175544);
    let r = i64x2::new(3860797565600356056, 8698062733717804468);

    assert_eq!(r, transmute(lsx_vneg_w(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vneg_d() {
    let a = i64x2::new(4241851098775470984, 2487122929432859927);
    let r = i64x2::new(-4241851098775470984, -2487122929432859927);

    assert_eq!(r, transmute(lsx_vneg_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmuh_b() {
    let a = i8x16::new(
        -123, 8, -7, 107, 85, 70, 44, 54, -34, -38, 48, 6, -23, 54, 25, -117,
    );
    let b = i8x16::new(
        41, -97, -9, -98, 27, 101, -95, 58, 102, -37, -72, -8, 94, -112, -22, -61,
    );
    let r = i64x2::new(931993372669836524, 2017024359980467698);

    assert_eq!(r, transmute(lsx_vmuh_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmuh_h() {
    let a = i16x8::new(-7394, -18356, -22999, 24389, 5841, 15177, -27319, -19905);
    let b = i16x8::new(-446, -16863, 19467, -13578, -9673, -26572, -7864, 9855);
    let r = i64x2::new(-1422322400225984462, -842721997477184351);

    assert_eq!(r, transmute(lsx_vmuh_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmuh_w() {
    let a = i32x4::new(1709346012, -2115891417, -530450121, 975457270);
    let b = i32x4::new(-1684820454, 449222301, 1106076122, 431017950);
    let r = i64x2::new(-950505610786872114, 420439596918869732);

    assert_eq!(r, transmute(lsx_vmuh_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmuh_d() {
    let a = i64x2::new(1852303942214142839, -864913423017390364);
    let b = i64x2::new(-1208434038665242614, -6078343251861677818);
    let r = i64x2::new(-121343209662433286, 284995587689374477);

    assert_eq!(r, transmute(lsx_vmuh_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmuh_bu() {
    let a = u8x16::new(
        7, 62, 97, 52, 145, 32, 36, 208, 81, 215, 70, 254, 95, 229, 130, 220,
    );
    let b = u8x16::new(
        220, 110, 97, 25, 127, 138, 167, 150, 128, 32, 130, 157, 177, 237, 123, 244,
    );
    let r = i64x2::new(8725461799780227590, -3369022092985820632);

    assert_eq!(r, transmute(lsx_vmuh_bu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmuh_hu() {
    let a = u16x8::new(28423, 34360, 7900, 61040, 62075, 6281, 10041, 37733);
    let b = u16x8::new(14769, 6489, 58866, 5997, 46648, 26325, 42186, 26942);
    let r = i64x2::new(1572068217944938757, 4366267597274655896);

    assert_eq!(r, transmute(lsx_vmuh_hu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmuh_wu() {
    let a = u32x4::new(1924935822, 3107975337, 289660636, 1367017690);
    let b = u32x4::new(1981234883, 1290836259, 1284878577, 702668871);
    let r = i64x2::new(4011887256539048298, 960560772888018584);

    assert_eq!(r, transmute(lsx_vmuh_wu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmuh_du() {
    let a = u64x2::new(11605461634325977288, 4587630571657223131);
    let b = u64x2::new(14805542397189366587, 10025341254588295994);
    let r = i64x2::new(-9132083796568587258, 2493261783600858707);

    assert_eq!(r, transmute(lsx_vmuh_du(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsllwil_h_b() {
    let a = i8x16::new(
        -45, 48, 102, -110, 126, -43, 65, 14, 75, 88, 62, 46, -109, 119, -77, 59,
    );
    let r = i64x2::new(-990777899147527584, 126109727303143360);

    assert_eq!(r, transmute(lsx_vsllwil_h_b::<5>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsllwil_w_h() {
    let a = i16x8::new(25135, -4241, 25399, -32451, 5597, -16847, 3192, -14694);
    let r = i64x2::new(-9326057613926912, -71360503652913664);

    assert_eq!(r, transmute(lsx_vsllwil_w_h::<9>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsllwil_d_w() {
    let a = i32x4::new(1472328927, -2106442262, 379100488, -607174188);
    let r = i64x2::new(6030659284992, -8627987505152);

    assert_eq!(r, transmute(lsx_vsllwil_d_w::<12>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsllwil_hu_bu() {
    let a = u8x16::new(
        102, 12, 222, 193, 16, 21, 161, 189, 127, 57, 231, 81, 97, 68, 171, 68,
    );
    let r = i64x2::new(6953679870551405312, 6809531147446388736);

    assert_eq!(r, transmute(lsx_vsllwil_hu_bu::<7>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsllwil_wu_hu() {
    let a = u16x8::new(370, 47410, 29611, 6206, 10390, 34658, 65264, 5264);
    let r = i64x2::new(52127846272954880, 6823569169558272);

    assert_eq!(r, transmute(lsx_vsllwil_wu_hu::<8>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsllwil_du_wu() {
    let a = u32x4::new(3249798491, 4098547305, 1101510259, 3478509641);
    let r = i64x2::new(13630642809995264, 17190553355550720);

    assert_eq!(r, transmute(lsx_vsllwil_du_wu::<22>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsran_b_h() {
    let a = i16x8::new(-12554, -869, 6838, -18394, -26140, 20902, -222, -12466);
    let b = i16x8::new(-12507, -16997, -17826, 5682, -298, -28572, -8117, -13478);
    let r = i64x2::new(-864943573596831881, 0);

    assert_eq!(r, transmute(lsx_vsran_b_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsran_h_w() {
    let a = i32x4::new(-950913431, 1557805031, 693572398, 1180916410);
    let b = i32x4::new(-52337348, -677553123, -58200260, -1473338606);
    let r = i64x2::new(1267763303694925820, 0);

    assert_eq!(r, transmute(lsx_vsran_h_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsran_w_d() {
    let a = i64x2::new(-1288554130833689959, -11977059487539737);
    let b = i64x2::new(-8585295495893484131, -2657141976436452013);
    let r = i64x2::new(-5882350952887806270, 0);

    assert_eq!(r, transmute(lsx_vsran_w_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssran_b_h() {
    let a = i16x8::new(-4232, -6038, -25131, -31144, -8955, 30109, -20875, 31748);
    let b = i16x8::new(9459, 15241, 22170, 28027, 5348, 14784, 22613, -9469);
    let r = i64x2::new(9187483431610086528, 0);

    assert_eq!(r, transmute(lsx_vssran_b_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssran_h_w() {
    let a = i32x4::new(-287861089, -1513011801, -2092611716, -303792243);
    let b = i32x4::new(2070726003, -944816867, -160621862, -1222036466);
    let r = i64x2::new(-5219109151313101350, 0);

    assert_eq!(r, transmute(lsx_vssran_h_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssran_w_d() {
    let a = i64x2::new(-3241370354549914429, -6946993314161316482);
    let b = i64x2::new(-7078666005882550400, -2564990402652718339);
    let r = i64x2::new(-15032385536, 0);

    assert_eq!(r, transmute(lsx_vssran_w_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssran_bu_h() {
    let a = u16x8::new(42413, 20386, 34692, 25088, 5477, 58748, 14986, 55598);
    let b = u16x8::new(2372, 26267, 4722, 47876, 44857, 55242, 45998, 51450);
    let r = i64x2::new(47227865344, 0);

    assert_eq!(r, transmute(lsx_vssran_bu_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssran_hu_w() {
    let a = u32x4::new(98545765, 1277336728, 1198651242, 2259455561);
    let b = u32x4::new(2085279153, 2679576985, 2935643238, 3797496208);
    let r = i64x2::new(281470684234479, 0);

    assert_eq!(r, transmute(lsx_vssran_hu_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssran_wu_d() {
    let a = u64x2::new(13769400838855917836, 9078517924805296472);
    let b = u64x2::new(3904652404244024971, 4230656884168675704);
    let r = i64x2::new(536870912000, 0);

    assert_eq!(r, transmute(lsx_vssran_wu_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrarn_b_h() {
    let a = i16x8::new(416, 1571, 19122, -32078, 26657, 3230, 12936, -5041);
    let b = i16x8::new(-19071, -903, 11542, -25909, 24111, 14882, -27192, -8283);
    let r = i64x2::new(7076043428318610384, 0);

    assert_eq!(r, transmute(lsx_vsrarn_b_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrarn_h_w() {
    let a = i32x4::new(-1553871953, -1700232136, 1934164676, -322997351);
    let b = i32x4::new(-1571698573, 1467958613, -1857488008, 424713310);
    let r = i64x2::new(498163119212, 0);

    assert_eq!(r, transmute(lsx_vsrarn_h_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrarn_w_d() {
    let a = i64x2::new(3489546309777968442, 4424654979674624573);
    let b = i64x2::new(-8645668865455529235, -3129277582817496880);
    let r = i64x2::new(-8628090759335017621, 0);

    assert_eq!(r, transmute(lsx_vsrarn_w_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrarn_b_h() {
    let a = i16x8::new(18764, -32156, 11073, -19939, -921, -18342, -16600, -13755);
    let b = i16x8::new(24298, 2343, 24641, 20910, 3142, -1171, 25850, 15932);
    let r = i64x2::new(-148338468081139694, 0);

    assert_eq!(r, transmute(lsx_vssrarn_b_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrarn_h_w() {
    let a = i32x4::new(-319370354, 225260835, 556195246, -699782233);
    let b = i32x4::new(1911424854, -931292983, -1710824608, -1179580317);
    let r = i64x2::new(-9223231301513904204, 0);

    assert_eq!(r, transmute(lsx_vssrarn_h_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrarn_w_d() {
    let a = i64x2::new(2645407519038125699, -6014465513887172991);
    let b = i64x2::new(2843689038926761304, -6830262024912907383);
    let r = i64x2::new(-9223372034707292161, 0);

    assert_eq!(r, transmute(lsx_vssrarn_w_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrarn_bu_h() {
    let a = u16x8::new(291, 64545, 16038, 57382, 18088, 10736, 57416, 55855);
    let b = u16x8::new(60210, 40155, 14296, 25577, 1550, 1674, 5330, 10645);
    let r = i64x2::new(10999415373897, 0);

    assert_eq!(r, transmute(lsx_vssrarn_bu_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrarn_hu_w() {
    let a = u32x4::new(2157227758, 1970326245, 1829195047, 4061259315);
    let b = u32x4::new(3570029841, 3229468238, 1070101998, 3159433736);
    let r = i64x2::new(281474976645120, 0);

    assert_eq!(r, transmute(lsx_vssrarn_hu_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrarn_wu_d() {
    let a = u64x2::new(8474558908443232483, 12352412821911429821);
    let b = u64x2::new(1112771813772164907, 646071836375127186);
    let r = i64x2::new(963446, 0);

    assert_eq!(r, transmute(lsx_vssrarn_wu_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrln_b_h() {
    let a = i16x8::new(11215, 29524, -2225, -13955, 13622, 15178, -22920, 29185);
    let b = i16x8::new(-11667, 13077, -23656, 5150, -23771, -31329, 20729, 15169);
    let r = i64x2::new(23363148983015937, 0);

    assert_eq!(r, transmute(lsx_vsrln_b_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrln_h_w() {
    let a = i32x4::new(273951092, 1016537129, 330941412, 1091816631);
    let b = i32x4::new(1775989751, -1602688801, -801213995, -1801759515);
    let r = i64x2::new(-7033214568759295968, 0);

    assert_eq!(r, transmute(lsx_vsrln_h_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrln_w_d() {
    let a = i64x2::new(-4929290425724370873, -9113314549902232460);
    let b = i64x2::new(-1428152872702150626, 3907864416256094744);
    let r = i64x2::new(-8718771486483115547, 0);

    assert_eq!(r, transmute(lsx_vsrln_w_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrln_bu_h() {
    let a = u16x8::new(53048, 1006, 61143, 41996, 57058, 25724, 43969, 62847);
    let b = u16x8::new(41072, 41125, 44619, 49581, 20733, 905, 47558, 7801);
    let r = i64x2::new(8862857593125412863, 0);

    assert_eq!(r, transmute(lsx_vssrln_bu_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrln_hu_w() {
    let a = u32x4::new(1889365848, 1818261427, 2701385771, 4063178210);
    let b = u32x4::new(1325069171, 1380839173, 3495604120, 2839043866);
    let r = i64x2::new(16889194387279379, 0);

    assert_eq!(r, transmute(lsx_vssrln_hu_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrln_wu_d() {
    let a = u64x2::new(7819967077464554342, 9878605573134710521);
    let b = u64x2::new(3908262745817581251, 17131627096934512209);
    let r = i64x2::new(-1, 0);

    assert_eq!(r, transmute(lsx_vssrln_wu_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrlrn_b_h() {
    let a = i16x8::new(-28299, -15565, -30638, -10884, -2538, 23256, 25217, 14524);
    let b = i16x8::new(22830, -27866, -24616, -9547, 11336, 320, 19908, 7056);
    let r = i64x2::new(-4888418841542521598, 0);

    assert_eq!(r, transmute(lsx_vsrlrn_b_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrlrn_h_w() {
    let a = i32x4::new(-146271143, 1373068571, 1580809863, -915867973);
    let b = i32x4::new(1387862348, 119424523, 185407104, 1890720739);
    let r = i64x2::new(2222313691660711041, 0);

    assert_eq!(r, transmute(lsx_vsrlrn_h_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrlrn_w_d() {
    let a = i64x2::new(-4585118244955419935, -6462467970618862820);
    let b = i64x2::new(-8550351213501194562, 7071641301481388656);
    let r = i64x2::new(182866822561795, 0);

    assert_eq!(r, transmute(lsx_vsrlrn_w_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlrn_bu_h() {
    let a = u16x8::new(13954, 8090, 46576, 53579, 4322, 20972, 17281, 18603);
    let b = u16x8::new(51122, 39148, 45511, 57479, 62603, 43668, 5537, 61004);
    let r = i64x2::new(432344477600776959, 0);

    assert_eq!(r, transmute(lsx_vssrlrn_bu_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlrn_hu_w() {
    let a = u32x4::new(959062112, 2073250884, 2500149644, 3919033303);
    let b = u32x4::new(1618795892, 3678356443, 862445734, 2115250342);
    let r = i64x2::new(-4293983341, 0);

    assert_eq!(r, transmute(lsx_vssrlrn_hu_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlrn_wu_d() {
    let a = u64x2::new(13828499145464267218, 4059850184169338184);
    let b = u64x2::new(13406765083608623828, 7214649593148131096);
    let r = i64x2::new(-1, 0);

    assert_eq!(r, transmute(lsx_vssrlrn_wu_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfrstpi_b() {
    let a = i8x16::new(
        116, 124, 21, 48, 24, 119, -108, 103, -77, -95, 68, -76, 67, -82, -96, 17,
    );
    let b = i8x16::new(
        -124, -52, -31, -108, 33, 71, -22, 0, -38, -20, -6, -90, 41, -58, -51, -51,
    );
    let r = i64x2::new(7463721428229389428, 1270206412966109619);

    assert_eq!(
        r,
        transmute(lsx_vfrstpi_b::<28>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfrstpi_h() {
    let a = i16x8::new(8411, -11473, 30045, -14781, 12135, -6534, -3622, 21173);
    let b = i16x8::new(9590, -8044, 15088, 4172, 1721, 27581, -19895, -25679);
    let r = i64x2::new(-4160352588467724069, 5959935604366651239);

    assert_eq!(r, transmute(lsx_vfrstpi_h::<1>(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfrstp_b() {
    let a = i8x16::new(
        41, -46, -4, 113, -42, 96, 62, 9, 12, -71, -82, 3, 4, -42, 43, -57,
    );
    let b = i8x16::new(
        -123, 108, -25, -29, -60, 41, -50, -93, 33, 99, 43, 36, 41, 88, 125, 27,
    );
    let c = i8x16::new(
        94, 2, 35, 33, 56, -117, -67, 85, 48, 94, -20, 112, -92, 47, -13, -80,
    );
    let r = i64x2::new(666076269049074217, -4107047547431896820);

    assert_eq!(
        r,
        transmute(lsx_vfrstp_b(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfrstp_h() {
    let a = i16x8::new(-23724, -17384, -24117, -29825, -19683, -3257, 18098, 7693);
    let b = i16x8::new(-20325, 3010, -32157, -32381, 13895, 10305, -4480, -12994);
    let c = i16x8::new(-2897, -31862, -29510, -16688, -12596, -6396, 20900, -22026);
    let r = i64x2::new(-8394813283989150892, 77734399685405);

    assert_eq!(
        r,
        transmute(lsx_vfrstp_h(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vshuf4i_d() {
    let a = i64x2::new(358242861525536259, -3448068840836542886);
    let b = i64x2::new(-5242415653399550268, -1504319281108156436);
    let r = i64x2::new(-3448068840836542886, -5242415653399550268);

    assert_eq!(
        r,
        transmute(lsx_vshuf4i_d::<153>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbsrl_v() {
    let a = i8x16::new(
        67, 57, -68, -24, 50, 58, 127, -80, -9, 17, 119, 81, 4, 110, 63, 56,
    );
    let r = i64x2::new(4570595419764160432, 56);

    assert_eq!(r, transmute(lsx_vbsrl_v::<7>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vbsll_v() {
    let a = i8x16::new(
        -25, -57, 97, -71, 66, 71, -127, 74, -32, -1, 36, 111, 116, 79, 49, -92,
    );
    let r = i64x2::new(0, -1801439850948198400);

    assert_eq!(r, transmute(lsx_vbsll_v::<15>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vextrins_b() {
    let a = i8x16::new(
        72, 112, -116, 99, 55, 19, 50, -123, -98, -90, 79, -29, 18, -87, 79, 74,
    );
    let b = i8x16::new(
        -107, 59, -127, 85, -65, -45, 80, 65, 30, -46, -56, -117, 107, 122, 11, -55,
    );
    let r = i64x2::new(-8848989189215300792, 5354684380554962590);

    assert_eq!(
        r,
        transmute(lsx_vextrins_b::<21>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vextrins_h() {
    let a = i16x8::new(-8903, 13698, -1855, 30429, -28178, 21171, -17068, -10547);
    let b = i16x8::new(-16309, 24895, 7753, 1535, 20205, 23989, 27706, -24274);
    let r = i64x2::new(8565108990437154105, -2968508409504886290);

    assert_eq!(
        r,
        transmute(lsx_vextrins_h::<33>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vextrins_w() {
    let a = i32x4::new(1225397826, 1289583478, 1287364839, 1276008188);
    let b = i32x4::new(1511106319, -1591171516, -989081993, 1462597836);
    let r = i64x2::new(5538718864697333314, -6834029622259375897);

    assert_eq!(
        r,
        transmute(lsx_vextrins_w::<57>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vextrins_d() {
    let a = i64x2::new(7112618873032505596, -3605623410483258197);
    let b = i64x2::new(-8508848216355653905, -4655572653097801607);
    let r = i64x2::new(7112618873032505596, -8508848216355653905);

    assert_eq!(
        r,
        transmute(lsx_vextrins_d::<62>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmskltz_b() {
    let a = i8x16::new(
        94, -6, -27, 108, 33, -86, -64, 68, 68, 9, -92, -83, -61, 99, 103, -77,
    );
    let r = i64x2::new(40038, 0);

    assert_eq!(r, transmute(lsx_vmskltz_b(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmskltz_h() {
    let a = i16x8::new(16730, 29121, -23447, -8647, -22303, 21817, 30964, -27069);
    let r = i64x2::new(156, 0);

    assert_eq!(r, transmute(lsx_vmskltz_h(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmskltz_w() {
    let a = i32x4::new(-657282776, -1247210048, 162595942, 949871015);
    let r = i64x2::new(3, 0);

    assert_eq!(r, transmute(lsx_vmskltz_w(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmskltz_d() {
    let a = i64x2::new(7728638770319849738, 4250984610820351699);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vmskltz_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsigncov_b() {
    let a = i8x16::new(
        37, -39, 115, 66, -114, -76, -55, -39, -94, 114, 38, 13, 76, 124, 64, -67,
    );
    let b = i8x16::new(
        -56, -98, -95, 45, 65, -53, -16, 126, 78, -69, -10, 115, -110, 125, -110, -27,
    );
    let r = i64x2::new(-9074694153930972472, 1986788453588057010);

    assert_eq!(r, transmute(lsx_vsigncov_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsigncov_h() {
    let a = i16x8::new(-2481, 28461, 27326, -11105, -17659, 25439, 5753, -743);
    let b = i16x8::new(27367, 4727, -2962, 14937, 26207, -19075, -26630, 10708);
    let r = i64x2::new(-4204122973533661927, -3013866947575178847);

    assert_eq!(r, transmute(lsx_vsigncov_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsigncov_w() {
    let a = i32x4::new(-1532048051, -2015529516, -586660708, 727735992);
    let b = i32x4::new(-1719915889, 290419288, 202835952, -1715336967);
    let r = i64x2::new(-1247341342367689359, -7367316170792699888);

    assert_eq!(r, transmute(lsx_vsigncov_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsigncov_d() {
    let a = i64x2::new(150793719457004094, -135856607031921617);
    let b = i64x2::new(-7146260093067324952, -4263419240070336957);
    let r = i64x2::new(-7146260093067324952, 4263419240070336957);

    assert_eq!(r, transmute(lsx_vsigncov_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfmadd_s() {
    let a = u32x4::new(1053592010, 1057663388, 1062706459, 1052867704);
    let b = u32x4::new(1058664483, 1064225083, 1063099591, 1054461138);
    let c = u32x4::new(1054468004, 1058982987, 1020391296, 1060092638);
    let r = i64x2::new(4580180050664125165, 4564646927777478184);

    assert_eq!(
        r,
        transmute(lsx_vfmadd_s(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfmadd_d() {
    let a = u64x2::new(4606327684689705003, 4598694159366762396);
    let b = u64x2::new(4605185255799132053, 4599088917574843416);
    let c = u64x2::new(4602818020827041428, 4603108774373140110);
    let r = i64x2::new(4608172630826345532, 4603863964483257995);

    assert_eq!(
        r,
        transmute(lsx_vfmadd_d(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfmsub_s() {
    let a = u32x4::new(1044400636, 1063313520, 1060460798, 1056994960);
    let b = u32x4::new(1016037632, 1057190051, 1042434224, 1054669464);
    let c = u32x4::new(1063213924, 1047859900, 1063932683, 1059194076);
    let r = i64x2::new(4492556612533126096, -4695805165913139817);

    assert_eq!(
        r,
        transmute(lsx_vfmsub_s(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfmsub_d() {
    let a = u64x2::new(4594815360286672212, 4596595309069193244);
    let b = u64x2::new(4603027383886900468, 4603059771165364192);
    let c = u64x2::new(4602620994011391758, 4604927875076111771);
    let r = i64x2::new(-4622272149514797982, -4619451105624653598);

    assert_eq!(
        r,
        transmute(lsx_vfmsub_d(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfnmadd_s() {
    let a = u32x4::new(1061642899, 1052761434, 1063541119, 1058091924);
    let b = u32x4::new(1044610040, 1047755448, 1062197759, 1051199080);
    let c = u32x4::new(1061915520, 1064953425, 1057353824, 1063041453);
    let r = i64x2::new(-4645363120071402583, -4645972958179775591);

    assert_eq!(
        r,
        transmute(lsx_vfnmadd_s(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfnmadd_d() {
    let a = u64x2::new(4581972604415454304, 4606375442608807393);
    let b = u64x2::new(4601574488118710932, 4600732882837014710);
    let c = u64x2::new(4598552045727299030, 4597905936756546488);
    let r = i64x2::new(-4624646832280694111, -4619798024319766060);

    assert_eq!(
        r,
        transmute(lsx_vfnmadd_d(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfnmsub_s() {
    let a = u32x4::new(1063347858, 1055637882, 1012264384, 1037368648);
    let b = u32x4::new(1054477234, 1065181074, 1060000965, 1061867853);
    let c = u32x4::new(1064036393, 1038991248, 1057711476, 1049339888);
    let r = i64x2::new(-4706852781727946153, 4486413029030305466);

    assert_eq!(
        r,
        transmute(lsx_vfnmsub_s(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfnmsub_d() {
    let a = u64x2::new(4604322037070318179, 4603593616949749938);
    let b = u64x2::new(4598988625246003058, 4600654731040688846);
    let c = u64x2::new(4601892672002082676, 4603822465490492305);
    let r = i64x2::new(4598264167668253799, 4600765330842720520);

    assert_eq!(
        r,
        transmute(lsx_vfnmsub_d(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrne_w_s() {
    let a = u32x4::new(1031214064, 1059673230, 1042813024, 1053602874);
    let r = i64x2::new(4294967296, 0);

    assert_eq!(r, transmute(lsx_vftintrne_w_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrne_l_d() {
    let a = u64x2::new(4606989588359571497, 4604713245380178790);
    let r = i64x2::new(1, 1);

    assert_eq!(r, transmute(lsx_vftintrne_l_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrp_w_s() {
    let a = u32x4::new(1061716225, 1050491008, 1064711040, 1065018777);
    let r = i64x2::new(4294967297, 4294967297);

    assert_eq!(r, transmute(lsx_vftintrp_w_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrp_l_d() {
    let a = u64x2::new(4587516915944025472, 4601504548481216392);
    let r = i64x2::new(1, 1);

    assert_eq!(r, transmute(lsx_vftintrp_l_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrm_w_s() {
    let a = u32x4::new(1045772456, 1065200707, 1061587478, 1035467272);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vftintrm_w_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrm_l_d() {
    let a = u64x2::new(4597123259408216804, 4594399417822716772);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vftintrm_l_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftint_w_d() {
    let a = u64x2::new(4602226310642310974, 4598315153561102162);
    let b = u64x2::new(4606905060326467647, 4606985586417166381);
    let r = i64x2::new(4294967297, 0);

    assert_eq!(r, transmute(lsx_vftint_w_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vffint_s_l() {
    let a = i64x2::new(-958368210120518642, 317739970300630807);
    let b = i64x2::new(5814449889729512723, -111756032377486319);
    let r = i64x2::new(-2610252963668467161, 6669016150524087533);

    assert_eq!(r, transmute(lsx_vffint_s_l(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrz_w_d() {
    let a = u64x2::new(4588311497244995104, 4604793095801710714);
    let b = u64x2::new(4599106720144900270, 4600531579473237336);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vftintrz_w_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrp_w_d() {
    let a = u64x2::new(4595926440353149184, 4601703964116560606);
    let b = u64x2::new(4606104970322966899, 4595679410565085836);
    let r = i64x2::new(4294967297, 4294967297);

    assert_eq!(r, transmute(lsx_vftintrp_w_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrm_w_d() {
    let a = u64x2::new(4603847521361653326, 4600607722530696016);
    let b = u64x2::new(4606733822200032543, 4589510164179968984);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vftintrm_w_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrne_w_d() {
    let a = u64x2::new(4601878512717779358, 4597694557130026508);
    let b = u64x2::new(4599197176714081204, 4605745859931721980);
    let r = i64x2::new(4294967296, 0);

    assert_eq!(r, transmute(lsx_vftintrne_w_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintl_l_s() {
    let a = u32x4::new(1058856635, 1060563398, 1061422616, 1056124918);
    let r = i64x2::new(1, 1);

    assert_eq!(r, transmute(lsx_vftintl_l_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftinth_l_s() {
    let a = u32x4::new(1045383680, 1040752748, 1061879518, 1054801708);
    let r = i64x2::new(1, 0);

    assert_eq!(r, transmute(lsx_vftinth_l_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vffinth_d_w() {
    let a = i32x4::new(517100418, -188510766, 949226647, -87467194);
    let r = i64x2::new(4741245898611228672, -4497729803343888384);

    assert_eq!(r, transmute(lsx_vffinth_d_w(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vffintl_d_w() {
    let a = i32x4::new(1273684401, -2137528906, -2109294912, -1646387998);
    let r = i64x2::new(4743129027571613696, -4476619782820462592);

    assert_eq!(r, transmute(lsx_vffintl_d_w(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrzl_l_s() {
    let a = u32x4::new(1031186688, 987838976, 1034565688, 1061017371);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vftintrzl_l_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrzh_l_s() {
    let a = u32x4::new(1049433828, 1048953580, 1060964637, 1059899586);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vftintrzh_l_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrpl_l_s() {
    let a = u32x4::new(1061834803, 1064858941, 1060475110, 1063896216);
    let r = i64x2::new(1, 1);

    assert_eq!(r, transmute(lsx_vftintrpl_l_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrph_l_s() {
    let a = u32x4::new(1059691939, 1065187151, 1059017027, 1061117394);
    let r = i64x2::new(1, 1);

    assert_eq!(r, transmute(lsx_vftintrph_l_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrml_l_s() {
    let a = u32x4::new(1062985651, 1065211455, 1056421466, 1057373572);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vftintrml_l_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrmh_l_s() {
    let a = u32x4::new(1050224290, 1063763666, 1057677270, 1063622234);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vftintrmh_l_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrnel_l_s() {
    let a = u32x4::new(1060174609, 1050974638, 1047193308, 1062040876);
    let r = i64x2::new(1, 0);

    assert_eq!(r, transmute(lsx_vftintrnel_l_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vftintrneh_l_s() {
    let a = u32x4::new(1055675382, 1036879184, 1064176794, 1063791852);
    let r = i64x2::new(1, 1);

    assert_eq!(r, transmute(lsx_vftintrneh_l_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfrintrne_s() {
    let a = u32x4::new(1054667842, 1061395025, 1062986478, 1062529334);
    let r = i64x2::new(4575657221408423936, 4575657222473777152);

    assert_eq!(r, transmute(lsx_vfrintrne_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfrintrne_d() {
    let a = u64x2::new(4603260356641870565, 4601614335120512898);
    let r = i64x2::new(4607182418800017408, 0);

    assert_eq!(r, transmute(lsx_vfrintrne_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfrintrz_s() {
    let a = u32x4::new(1063039577, 1033416832, 1052369306, 1057885024);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfrintrz_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfrintrz_d() {
    let a = u64x2::new(4601515428088814484, 4604735152905786794);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfrintrz_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfrintrp_s() {
    let a = u32x4::new(1061968959, 1056597596, 1064869916, 1058742360);
    let r = i64x2::new(4575657222473777152, 4575657222473777152);

    assert_eq!(r, transmute(lsx_vfrintrp_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfrintrp_d() {
    let a = u64x2::new(4603531792479663401, 4587997630530425392);
    let r = i64x2::new(4607182418800017408, 4607182418800017408);

    assert_eq!(r, transmute(lsx_vfrintrp_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfrintrm_s() {
    let a = u32x4::new(1058024441, 1044087184, 1059777964, 1050835426);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfrintrm_s(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfrintrm_d() {
    let a = u64x2::new(4589388034824743512, 4606800774570289382);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfrintrm_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vstelm_b() {
    let a = i8x16::new(
        -70, -74, -13, -53, -37, -28, -84, -8, 110, -98, -26, 71, 55, 104, -8, -50,
    );
    let mut o: [i8; 16] = [
        97, 16, 51, -123, 4, 14, 108, 36, -40, -53, 29, 67, 102, 63, -15, -39,
    ];
    let r = i64x2::new(2624488095427530938, -2742340989646681128);

    lsx_vstelm_b::<0, 0>(transmute(a), o.as_mut_ptr());
    assert_eq!(r, transmute(o));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vstelm_h() {
    let a = i16x8::new(-7427, -5749, 19902, -9799, 28691, -16170, 11920, 24129);
    let mut o: [i8; 16] = [
        123, 19, -3, 118, -43, -40, -48, -81, 23, -114, -72, 26, 117, 98, -43, -112,
    ];
    let r = i64x2::new(-5777879910580360821, -8010388107109560809);

    lsx_vstelm_h::<0, 1>(transmute(a), o.as_mut_ptr());
    assert_eq!(r, transmute(o));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vstelm_w() {
    let a = i32x4::new(424092909, 1956922334, -640221305, -164680666);
    let mut o: [i8; 16] = [
        -12, -50, 8, 91, 60, -48, 94, -99, -64, -51, 3, -44, 7, -49, 62, -69,
    ];
    let r = i64x2::new(-7107014201697162202, -4954294907532227136);

    lsx_vstelm_w::<0, 3>(transmute(a), o.as_mut_ptr());
    assert_eq!(r, transmute(o));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vstelm_d() {
    let a = i64x2::new(2628828971609511929, 9138529437562240974);
    let mut o: [i8; 16] = [
        48, -98, 127, -32, 90, 120, 50, 2, 90, 120, -113, 19, -120, 105, 27, -22,
    ];
    let r = i64x2::new(2628828971609511929, -1577551211298588582);

    lsx_vstelm_d::<0, 0>(transmute(a), o.as_mut_ptr());
    assert_eq!(r, transmute(o));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwev_d_w() {
    let a = i32x4::new(-1889902301, 326462140, 1088579813, 626337726);
    let b = i32x4::new(-2105551735, -1478351177, 1027048582, -607110700);
    let r = i64x2::new(-3995454036, 2115628395);

    assert_eq!(r, transmute(lsx_vaddwev_d_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwev_w_h() {
    let a = i16x8::new(7813, 337, -10949, -8624, 14298, -27002, -12747, 17169);
    let b = i16x8::new(-17479, -32614, 24343, 25426, -14077, -12419, 10115, 23013);
    let r = i64x2::new(57531086920254, -11304353922851);

    assert_eq!(r, transmute(lsx_vaddwev_w_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwev_h_b() {
    let a = i8x16::new(
        -122, -50, 126, -108, 72, 89, -50, -96, -37, -68, 63, -41, -1, -49, 90, 117,
    );
    let b = i8x16::new(
        -89, 6, -27, 58, 80, -29, 28, 104, 30, 69, -39, 76, 42, 34, 25, -24,
    );
    let r = i64x2::new(-6191796646052051, 32369798417022969);

    assert_eq!(r, transmute(lsx_vaddwev_h_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwod_d_w() {
    let a = i32x4::new(-1721333318, -347227654, -936088440, 1975890670);
    let b = i32x4::new(420515981, 473447119, 1471756335, 1044924117);
    let r = i64x2::new(126219465, 3020814787);

    assert_eq!(r, transmute(lsx_vaddwod_d_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwod_w_h() {
    let a = i16x8::new(13058, 5020, 31112, -31710, 19542, -9009, -21764, -1881);
    let b = i16x8::new(-26581, -22301, 18214, -3616, -24489, 12150, -10765, -24232);
    let r = i64x2::new(-151719719748481, -112154480997307);

    assert_eq!(r, transmute(lsx_vaddwod_w_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwod_h_b() {
    let a = i8x16::new(
        -53, 61, 10, -18, -31, 26, 113, -14, -62, 6, 127, -43, 86, 33, 94, 57,
    );
    let b = i8x16::new(
        37, 85, -14, -93, 61, -116, -53, -51, -46, 119, 36, -94, 0, -86, 46, -6,
    );
    let r = i64x2::new(-18014780768845678, 14636475441676413);

    assert_eq!(r, transmute(lsx_vaddwod_h_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwev_d_wu() {
    let a = u32x4::new(2539947230, 3548211150, 1193982195, 3547334418);
    let b = u32x4::new(1482213353, 1001198416, 3345983326, 2244256337);
    let r = i64x2::new(4022160583, 4539965521);

    assert_eq!(r, transmute(lsx_vaddwev_d_wu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwev_w_hu() {
    let a = u16x8::new(50844, 55931, 31330, 63416, 32884, 2778, 22874, 13540);
    let b = u16x8::new(28483, 24704, 9817, 62062, 47674, 8032, 29897, 62737);
    let r = i64x2::new(176725019407839, 226649719257774);

    assert_eq!(r, transmute(lsx_vaddwev_w_hu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwev_h_bu() {
    let a = u8x16::new(
        233, 165, 29, 130, 62, 173, 207, 120, 32, 254, 152, 27, 30, 159, 92, 76,
    );
    let b = u8x16::new(
        118, 157, 181, 79, 81, 38, 95, 73, 245, 179, 126, 210, 16, 93, 78, 63,
    );
    let r = i64x2::new(85006057160704351, 47850943627526421);

    assert_eq!(r, transmute(lsx_vaddwev_h_bu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwod_d_wu() {
    let a = u32x4::new(342250989, 1651153980, 174227274, 2092816321);
    let b = u32x4::new(2782520439, 2496077290, 2678772394, 196273109);
    let r = i64x2::new(4147231270, 2289089430);

    assert_eq!(r, transmute(lsx_vaddwod_d_wu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwod_w_hu() {
    let a = u16x8::new(36372, 35690, 49187, 14265, 54130, 40094, 57017, 10670);
    let b = u16x8::new(20353, 34039, 21222, 4948, 58293, 4766, 51360, 37497);
    let r = i64x2::new(82519206727777, 206875689791292);

    assert_eq!(r, transmute(lsx_vaddwod_w_hu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwod_h_bu() {
    let a = u8x16::new(
        248, 1, 83, 240, 60, 173, 151, 39, 55, 39, 131, 86, 86, 18, 5, 110,
    );
    let b = u8x16::new(
        63, 52, 164, 249, 242, 167, 236, 222, 171, 180, 249, 57, 79, 53, 87, 7,
    );
    let r = i64x2::new(73466429242409013, 32932877227196635);

    assert_eq!(r, transmute(lsx_vaddwod_h_bu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwev_d_wu_w() {
    let a = u32x4::new(3787058271, 4254502892, 1291509641, 2971162106);
    let b = i32x4::new(-1308530150, 1427930358, 1723198474, 1987356336);
    let r = i64x2::new(2478528121, 3014708115);

    assert_eq!(r, transmute(lsx_vaddwev_d_wu_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwev_w_hu_h() {
    let a = u16x8::new(7742, 2564, 7506, 3394, 6835, 41043, 29153, 7959);
    let b = i16x8::new(-11621, -6593, 7431, -1189, -12361, -15174, 16182, -32434);
    let r = i64x2::new(64158221463769, 194716637325930);

    assert_eq!(r, transmute(lsx_vaddwev_w_hu_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwev_h_bu_b() {
    let a = u8x16::new(
        103, 224, 71, 251, 48, 94, 188, 16, 181, 57, 192, 250, 248, 36, 51, 176,
    );
    let b = i8x16::new(
        36, -32, 108, -95, -21, 20, 67, -107, -65, -124, -19, -50, -120, -36, -79, -12,
    );
    let r = i64x2::new(71776235037065355, -7880749580746636);

    assert_eq!(r, transmute(lsx_vaddwev_h_bu_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwod_d_wu_w() {
    let a = u32x4::new(3763905902, 2910980290, 1912906409, 2257280339);
    let b = i32x4::new(-1646368557, 586112311, 376247963, 1048800083);
    let r = i64x2::new(3497092601, 3306080422);

    assert_eq!(r, transmute(lsx_vaddwod_d_wu_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwod_w_hu_h() {
    let a = u16x8::new(53495, 36399, 39536, 12468, 17601, 52919, 14730, 58963);
    let b = i16x8::new(31700, 22725, 14068, -14860, -28839, -14513, -1195, 27082);
    let r = i64x2::new(-10273561712908, 369560461022726);

    assert_eq!(r, transmute(lsx_vaddwod_w_hu_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwod_h_bu_b() {
    let a = u8x16::new(
        191, 183, 244, 200, 83, 191, 111, 82, 210, 150, 228, 182, 45, 23, 145, 159,
    );
    let b = i8x16::new(
        -34, -59, -104, -58, -78, 90, -117, 93, 76, -23, 37, 44, -62, 60, 119, -91,
    );
    let r = i64x2::new(49259327819481212, 19140654913421439);

    assert_eq!(r, transmute(lsx_vaddwod_h_bu_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsubwev_d_w() {
    let a = i32x4::new(1979919903, -1490022083, -1106776488, 2132235386);
    let b = i32x4::new(-2090701374, 629564229, -1170676885, 1069800209);
    let r = i64x2::new(4070621277, 63900397);

    assert_eq!(r, transmute(lsx_vsubwev_d_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsubwev_w_h() {
    let a = i16x8::new(1153, -17319, 23560, 30758, -11540, -15757, -5844, -31417);
    let b = i16x8::new(-23957, 9416, -29569, -13210, 5333, 8420, 18648, -24201);
    let r = i64x2::new(228187317494294, -105188044063209);

    assert_eq!(r, transmute(lsx_vsubwev_w_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsubwev_h_b() {
    let a = i8x16::new(
        123, 120, -48, 33, 4, -108, -68, -59, 54, 30, 17, -104, -30, -76, -127, -108,
    );
    let b = i8x16::new(
        -16, 108, -113, 37, -118, 72, 81, 103, 63, -86, -109, -71, -29, 83, -75, 97,
    );
    let r = i64x2::new(-41939247539617653, -14355228098887689);

    assert_eq!(r, transmute(lsx_vsubwev_h_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsubwod_d_w() {
    let a = i32x4::new(-1024625027, -1083407596, 1367079411, 1458097720);
    let b = i32x4::new(1436617964, -45524609, 502994793, -2039550077);
    let r = i64x2::new(-1037882987, 3497647797);

    assert_eq!(r, transmute(lsx_vsubwod_d_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsubwod_w_h() {
    let a = i16x8::new(-15137, 29913, 8889, -17237, 31133, 28017, 9070, -18477);
    let b = i16x8::new(-1276, 12669, 24115, 19617, -26739, 1910, -757, 23994);
    let r = i64x2::new(-158286724709540, -182411556002309);

    assert_eq!(r, transmute(lsx_vsubwod_w_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsubwod_h_b() {
    let a = i8x16::new(
        -25, -19, -117, -1, 9, 24, -16, 93, 9, -77, -36, 75, 0, 126, 74, -106,
    );
    let b = i8x16::new(
        -91, -3, -112, 5, -88, -14, -1, 8, -100, 65, -26, -24, 41, 124, 17, -108,
    );
    let r = i64x2::new(23925540523802608, 562958549909362);

    assert_eq!(r, transmute(lsx_vsubwod_h_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsubwev_d_wu() {
    let a = u32x4::new(2665672710, 2360377198, 3032815602, 1049776563);
    let b = u32x4::new(1691253880, 1939268473, 1629937431, 2921768539);
    let r = i64x2::new(974418830, 1402878171);

    assert_eq!(r, transmute(lsx_vsubwev_d_wu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsubwev_w_hu() {
    let a = u16x8::new(8298, 25954, 33403, 10264, 36066, 64035, 18750, 26396);
    let b = u16x8::new(15957, 42770, 43138, 30319, 50823, 18089, 64120, 18054);
    let r = i64x2::new(-41807211666923, -194858371266981);

    assert_eq!(r, transmute(lsx_vsubwev_w_hu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsubwev_h_bu() {
    let a = u8x16::new(
        128, 1, 20, 37, 75, 38, 156, 224, 7, 26, 190, 76, 144, 59, 175, 99,
    );
    let b = u8x16::new(
        141, 113, 141, 61, 31, 32, 161, 158, 220, 37, 240, 180, 56, 229, 5, 26,
    );
    let r = i64x2::new(-1407181617889293, 47851128289689387);

    assert_eq!(r, transmute(lsx_vsubwev_h_bu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsubwod_d_wu() {
    let a = u32x4::new(623751944, 3506098576, 826539449, 2248804942);
    let b = u32x4::new(103354715, 19070238, 1662532733, 3761231766);
    let r = i64x2::new(3487028338, -1512426824);

    assert_eq!(r, transmute(lsx_vsubwod_d_wu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsubwod_w_hu() {
    let a = u16x8::new(2891, 21215, 21876, 42023, 37208, 16456, 2023, 54703);
    let b = u16x8::new(21739, 45406, 21733, 63910, 6659, 16020, 1211, 637);
    let r = i64x2::new(-93999654264447, 232211701825972);

    assert_eq!(r, transmute(lsx_vsubwod_w_hu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsubwod_h_bu() {
    let a = u8x16::new(
        6, 39, 26, 92, 204, 140, 65, 76, 214, 200, 24, 203, 215, 17, 22, 226,
    );
    let b = u8x16::new(
        89, 14, 101, 173, 231, 124, 106, 127, 125, 115, 109, 27, 121, 175, 229, 175,
    );
    let r = i64x2::new(-14355150803107815, 14636020195655765);

    assert_eq!(r, transmute(lsx_vsubwod_h_bu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwev_q_d() {
    let a = i64x2::new(-1132117278547342347, -8844779319945501636);
    let b = i64x2::new(6738886902337351868, -5985538541381931477);
    let r = i64x2::new(5606769623790009521, 0);

    assert_eq!(r, transmute(lsx_vaddwev_q_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwod_q_d() {
    let a = i64x2::new(-8159683400941020659, -1142783567808544783);
    let b = i64x2::new(-1244049724346527963, -3275029038845457041);
    let r = i64x2::new(-4417812606654001824, -1);

    assert_eq!(r, transmute(lsx_vaddwod_q_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwev_q_du() {
    let a = u64x2::new(16775220860485391359, 8922486068170257729);
    let b = u64x2::new(6745766838534849346, 15041258018068294402);
    let r = i64x2::new(5074243625310689089, 1);

    assert_eq!(r, transmute(lsx_vaddwev_q_du(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwod_q_du() {
    let a = u64x2::new(17311013772674153390, 11698682577513574290);
    let b = u64x2::new(13496765248439164553, 4640846570780442359);
    let r = i64x2::new(-2107214925415534967, 0);

    assert_eq!(r, transmute(lsx_vaddwod_q_du(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsubwev_q_d() {
    let a = i64x2::new(8509296067394123199, 4972040966127046151);
    let b = i64x2::new(8029026411722387723, -2105201823388787841);
    let r = i64x2::new(480269655671735476, 0);

    assert_eq!(r, transmute(lsx_vsubwev_q_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsubwod_q_d() {
    let a = i64x2::new(-5518792681032609552, -5818770921355494107);
    let b = i64x2::new(5758437127240728961, 2933507971643343184);
    let r = i64x2::new(-8752278892998837291, -1);

    assert_eq!(r, transmute(lsx_vsubwod_q_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsubwev_q_du() {
    let a = u64x2::new(15348090063574162992, 4054607174208637377);
    let b = u64x2::new(1574118313456291324, 7787456577305510529);
    let r = i64x2::new(-4672772323591679948, 0);

    assert_eq!(r, transmute(lsx_vsubwev_q_du(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsubwod_q_du() {
    let a = u64x2::new(7199085452795040192, 586057639195920839);
    let b = u64x2::new(5627376085113520030, 12775637764770549815);
    let r = i64x2::new(6257163948134922640, -1);

    assert_eq!(r, transmute(lsx_vsubwod_q_du(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwev_q_du_d() {
    let a = u64x2::new(11103722789624608070, 8912888508651245205);
    let b = i64x2::new(-1159499132550683978, -4257322329662100669);
    let r = i64x2::new(-8502520416635627524, 0);

    assert_eq!(r, transmute(lsx_vaddwev_q_du_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vaddwod_q_du_d() {
    let a = u64x2::new(8904095231861536434, 126069624822744729);
    let b = i64x2::new(-3902573037873546881, 160140233311333524);
    let r = i64x2::new(286209858134078253, 0);

    assert_eq!(r, transmute(lsx_vaddwod_q_du_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwev_d_w() {
    let a = i32x4::new(1287102156, 1220933948, 1816088643, -266313269);
    let b = i32x4::new(8741677, -276509855, -1214560052, -1338519080);
    let r = i64x2::new(11251431313755612, -2205748716678689436);

    assert_eq!(r, transmute(lsx_vmulwev_d_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwev_w_h() {
    let a = i16x8::new(6427, -15587, -29266, -12748, 29941, -16072, -3936, -4131);
    let b = i16x8::new(30661, -20472, 1422, -16868, 4256, 9713, -27765, -7287);
    let r = i64x2::new(-178740441125036345, 469367082934888736);

    assert_eq!(r, transmute(lsx_vmulwev_w_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwev_h_b() {
    let a = i8x16::new(
        -53, -116, -37, -91, -27, -23, 3, -103, -83, 88, 61, -1, 37, 89, -77, -78,
    );
    let b = i8x16::new(
        102, -8, -8, -115, -104, 126, 46, 69, -53, 81, -41, 100, -83, -42, -38, -17,
    );
    let r = i64x2::new(38855607073696482, 823864071118590255);

    assert_eq!(r, transmute(lsx_vmulwev_h_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwod_d_w() {
    let a = i32x4::new(730217708, -1124949962, -360746398, -1749502167);
    let b = i32x4::new(63312847, -1377579771, -2054819244, -1416520586);
    let r = i64x2::new(1549708311038418702, 2478205834807109862);

    assert_eq!(r, transmute(lsx_vmulwod_d_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwod_w_h() {
    let a = i16x8::new(-16507, -11588, -4739, -32549, -22878, 5561, -6134, -3022);
    let b = i16x8::new(23748, 11912, 4946, -23048, 22372, 24702, -24875, -27771);
    let r = i64x2::new(3222038736804363232, 360450672278114574);

    assert_eq!(r, transmute(lsx_vmulwod_w_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwod_h_b() {
    let a = i8x16::new(
        -110, 22, -19, -91, 6, 25, -7, 13, 86, -110, -98, -100, -18, -111, 100, 31,
    );
    let b = i8x16::new(
        102, 16, -43, -24, -28, 2, 5, -96, 26, 74, -56, 109, -30, 40, -96, 109,
    );
    let r = i64x2::new(-351280556043402912, 951366355207905332);

    assert_eq!(r, transmute(lsx_vmulwod_h_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwev_d_wu() {
    let a = u32x4::new(2063305123, 761682812, 3318081558, 2848424479);
    let b = u32x4::new(1769900227, 2256955703, 2342391995, 2407560006);
    let r = i64x2::new(3651844205567962921, 7772247680216328210);

    assert_eq!(r, transmute(lsx_vmulwev_d_wu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwev_w_hu() {
    let a = u16x8::new(9553, 49381, 46053, 13610, 17063, 24513, 41196, 11695);
    let b = u16x8::new(20499, 45056, 20580, 12771, 53914, 60742, 45402, 40547);
    let r = i64x2::new(4070644332601545987, 8033224333626513014);

    assert_eq!(r, transmute(lsx_vmulwev_w_hu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwev_h_bu() {
    let a = u8x16::new(
        227, 157, 43, 90, 6, 141, 46, 1, 92, 129, 254, 35, 161, 83, 40, 101,
    );
    let b = u8x16::new(
        111, 233, 206, 13, 205, 128, 21, 105, 114, 77, 138, 243, 4, 51, 173, 180,
    );
    let r = i64x2::new(271910110892810861, 1947809607093856504);

    assert_eq!(r, transmute(lsx_vmulwev_h_bu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwod_d_wu() {
    let a = u32x4::new(2178610550, 1983075871, 1118106927, 2182535205);
    let b = u32x4::new(3750239707, 1422851626, 1277923597, 1377279439);
    let r = i64x2::new(2821622727533716246, 3005960862740149995);

    assert_eq!(r, transmute(lsx_vmulwod_d_wu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwod_w_hu() {
    let a = u16x8::new(63169, 54563, 40593, 32351, 22785, 46152, 51840, 54366);
    let b = u16x8::new(38950, 5357, 36233, 17707, 61077, 61518, 5789, 13317);
    let r = i64x2::new(2460325445475503463, 3109522059894091248);

    assert_eq!(r, transmute(lsx_vmulwod_w_hu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwod_h_bu() {
    let a = u8x16::new(
        143, 18, 19, 120, 134, 160, 86, 206, 25, 26, 241, 198, 207, 50, 233, 169,
    );
    let b = u8x16::new(
        244, 115, 210, 167, 103, 242, 182, 127, 214, 208, 47, 86, 54, 81, 161, 139,
    );
    let r = i64x2::new(7364114643151226902, 6612146073643521312);

    assert_eq!(r, transmute(lsx_vmulwod_h_bu(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwev_d_wu_w() {
    let a = u32x4::new(1829687775, 3948847254, 3506011389, 2834786083);
    let b = i32x4::new(1254729285, 1938836163, -1902169358, -257980375);
    let r = i64x2::new(2295762833698990875, -6669027432954818262);

    assert_eq!(r, transmute(lsx_vmulwev_d_wu_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwev_w_hu_h() {
    let a = u16x8::new(50708, 48173, 47753, 19808, 25837, 56376, 50749, 8070);
    let b = i16x8::new(-30477, -10049, 16428, -30668, 21000, 24834, -3219, -9555);
    let r = i64x2::new(3369342936690107644, -701630285043265176);

    assert_eq!(r, transmute(lsx_vmulwev_w_hu_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwev_h_bu_b() {
    let a = u8x16::new(
        196, 15, 88, 70, 49, 17, 144, 62, 34, 164, 51, 69, 162, 88, 100, 31,
    );
    let b = i8x16::new(
        -92, 119, 90, -113, -83, 119, -28, -14, 57, 93, -21, -38, 42, -105, -67, -73,
    );
    let r = i64x2::new(-1134643098233554544, -1885853116779133038);

    assert_eq!(r, transmute(lsx_vmulwev_h_bu_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwod_d_wu_w() {
    let a = u32x4::new(3252247725, 3029105766, 3286505645, 1763684728);
    let b = i32x4::new(1204047391, -1970001586, 608763444, -2082771896);
    let r = i64x2::new(-5967343163181744876, -3673352984882804288);

    assert_eq!(r, transmute(lsx_vmulwod_d_wu_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwod_w_hu_h() {
    let a = u16x8::new(38405, 41959, 20449, 33265, 58814, 59003, 64929, 20835);
    let b = i16x8::new(-3735, -12972, -4920, 7170, 11577, 9785, 4896, -537);
    let r = i64x2::new(1024392868267999948, -48053790042385565);

    assert_eq!(r, transmute(lsx_vmulwod_w_hu_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwod_h_bu_b() {
    let a = u8x16::new(
        78, 246, 141, 207, 212, 16, 30, 141, 71, 187, 92, 123, 199, 224, 105, 250,
    );
    let b = i8x16::new(
        46, 11, 86, 64, -118, -53, 125, 48, -122, 104, 53, -111, 39, 16, -94, -56,
    );
    let r = i64x2::new(1905300476090387090, -3940634277386171400);

    assert_eq!(r, transmute(lsx_vmulwod_h_bu_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwev_q_d() {
    let a = i64x2::new(-7300892474466935547, -2126323416087979991);
    let b = i64x2::new(7023560313675997328, 4368639658790376608);
    let r = i64x2::new(-1409563343912029488, -2779799970834089134);

    assert_eq!(r, transmute(lsx_vmulwev_q_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwod_q_d() {
    let a = i64x2::new(-333821925237206080, -2872872657001472243);
    let b = i64x2::new(1734538850547798281, 6505001633960390309);
    let r = i64x2::new(655114704133495137, -1013080750363369114);

    assert_eq!(r, transmute(lsx_vmulwod_q_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwev_q_du() {
    let a = u64x2::new(7574912843445409775, 6458810692359816933);
    let b = u64x2::new(15048173707940873365, 13594773395779002998);
    let r = i64x2::new(-4049323972691826149, 6179334620527225413);

    assert_eq!(r, transmute(lsx_vmulwev_q_du(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwod_q_du() {
    let a = u64x2::new(4945250618288414185, 5836523005600515765);
    let b = u64x2::new(16172423495582959833, 11676106279348566952);
    let r = i64x2::new(-66293137947075128, 3694303051148166412);

    assert_eq!(r, transmute(lsx_vmulwod_q_du(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwev_q_du_d() {
    let a = u64x2::new(15472635927451755137, 2872062649560660647);
    let b = i64x2::new(-7071166739782294817, 8496829998090419991);
    let r = i64x2::new(5234431817964974175, -5931105679667820544);

    assert_eq!(r, transmute(lsx_vmulwev_q_du_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmulwod_q_du_d() {
    let a = u64x2::new(2980498025260165803, 6347157252532266677);
    let b = i64x2::new(-9085162554263782091, -3351642387065053502);
    let r = i64x2::new(-3119502026085414102, -1153233394465180223);

    assert_eq!(r, transmute(lsx_vmulwod_q_du_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vhaddw_q_d() {
    let a = i64x2::new(-7668184096931639781, -2784020394780249366);
    let b = i64x2::new(9222966760421493517, -8347454331188625422);
    let r = i64x2::new(6438946365641244151, 0);

    assert_eq!(r, transmute(lsx_vhaddw_q_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vhaddw_qu_du() {
    let a = u64x2::new(16989728354409608690, 2941626047560944845);
    let b = u64x2::new(2141387370256045519, 12417156199252644485);
    let r = i64x2::new(5083013417816990364, 0);

    assert_eq!(r, transmute(lsx_vhaddw_qu_du(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vhsubw_q_d() {
    let a = i64x2::new(4415650624918824808, -2427685530964051137);
    let b = i64x2::new(-3245503809142406078, 8660213762027125085);
    let r = i64x2::new(817818278178354941, 0);

    assert_eq!(r, transmute(lsx_vhsubw_q_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vhsubw_qu_du() {
    let a = u64x2::new(13300663635362906510, 12554343611316218179);
    let b = u64x2::new(3098179646743711521, 11374525358855478565);
    let r = i64x2::new(-8990580109137044958, 0);

    assert_eq!(r, transmute(lsx_vhsubw_qu_du(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwev_d_w() {
    let a = i64x2::new(7507491558224723369, 7356288879446926343);
    let b = i32x4::new(-1410295112, 176083487, 1092174685, 1464381516);
    let c = i32x4::new(1610457028, -1105361927, -790658106, -1804307944);
    let r = i64x2::new(5236271883550276233, 6492752111583679733);

    assert_eq!(
        r,
        transmute(lsx_vmaddwev_d_w(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwev_w_h() {
    let a = i32x4::new(1210747897, 1541928975, -720014144, -2019635451);
    let b = i16x8::new(12181, 16380, -24682, -13729, 12128, -21312, -23449, 17);
    let c = i16x8::new(-27087, 21294, 30093, 5456, 28491, -25365, -18595, 14478);
    let r = i64x2::new(3432424257664054654, -6801515772302723616);

    assert_eq!(
        r,
        transmute(lsx_vmaddwev_w_h(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwev_h_b() {
    let a = i16x8::new(-26961, 27058, -26746, 7019, 27143, -20720, 20159, -22095);
    let b = i8x16::new(
        126, 29, -29, 63, -17, 109, 56, 67, 91, -76, 83, -101, 51, 39, -109, 16,
    );
    let c = i8x16::new(
        -40, -36, -53, -47, -78, 33, -97, -54, 21, 103, 69, 101, 33, -83, 79, -6,
    );
    let r = i64x2::new(446873086821892863, -8642876820889308802);

    assert_eq!(
        r,
        transmute(lsx_vmaddwev_h_b(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwev_d_wu() {
    let a = u64x2::new(3288783601225499701, 17730813816531737481);
    let b = u32x4::new(2583154680, 1751994654, 1115446691, 3761972534);
    let c = u32x4::new(1143913546, 2487138808, 577997991, 917071165);
    let r = i64x2::new(6243689231090794981, -71204310712216354);

    assert_eq!(
        r,
        transmute(lsx_vmaddwev_d_wu(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwev_w_hu() {
    let a = u32x4::new(805734379, 3876931235, 2135371653, 3482539797);
    let b = u16x8::new(7507, 65354, 30738, 63434, 34178, 38533, 8774, 9013);
    let c = u16x8::new(32752, 10153, 5275, 7485, 55213, 62803, 43040, 42218);
    let r = i64x2::new(-1099052541965094213, -1867428321461954977);

    assert_eq!(
        r,
        transmute(lsx_vmaddwev_w_hu(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwev_h_bu() {
    let a = u16x8::new(55814, 6276, 42400, 55862, 19175, 17360, 30132, 17253);
    let b = u8x16::new(
        148, 50, 79, 199, 193, 25, 144, 93, 18, 182, 102, 150, 226, 222, 254, 1,
    );
    let c = u8x16::new(
        141, 28, 169, 93, 60, 134, 117, 80, 43, 12, 75, 85, 174, 176, 62, 94,
    );
    let r = i64x2::new(2019533326543170442, -9157771529370317331);

    assert_eq!(
        r,
        transmute(lsx_vmaddwev_h_bu(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwod_d_w() {
    let a = i64x2::new(1296033816549937177, -2404834118264545479);
    let b = i32x4::new(-2135765262, -1741194198, -1750008434, -242816495);
    let c = i32x4::new(178412146, 887047455, -1630315539, 57253350);
    let r = i64x2::new(-248488065446728913, -2418736176038553729);

    assert_eq!(
        r,
        transmute(lsx_vmaddwod_d_w(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwod_w_h() {
    let a = i32x4::new(1810262555, -720984423, 744322940, -172229387);
    let b = i16x8::new(27584, -15468, -21544, -11891, -16682, 18538, -7573, -1522);
    let c = i16x8::new(-8815, 3268, -32219, -7020, 13853, 26700, -2030, -5667);
    let r = i64x2::new(-2738082894011230357, -702674743083530508);

    assert_eq!(
        r,
        transmute(lsx_vmaddwod_w_h(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwod_h_b() {
    let a = i16x8::new(32731, -16929, 397, 14417, 22494, 1416, 1669, -12175);
    let b = i8x16::new(
        87, 77, -44, -128, -69, 120, 82, -99, -21, 66, -47, -59, -35, 90, -85, 94,
    );
    let c = i8x16::new(
        87, -119, -48, 10, 26, -36, 89, -16, 91, -74, -116, 7, 78, 17, -9, -98,
    );
    let r = i64x2::new(4504145731268860944, -6019891587244669750);

    assert_eq!(
        r,
        transmute(lsx_vmaddwod_h_b(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwod_d_wu() {
    let a = u64x2::new(8272899369384595612, 11592257149528470828);
    let b = u32x4::new(244745450, 2190106289, 660562971, 1842569843);
    let c = u32x4::new(388973541, 2963125445, 520938623, 340863345);
    let r = i64x2::new(-3684285032134532399, -6226422404099975953);

    assert_eq!(
        r,
        transmute(lsx_vmaddwod_d_wu(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwod_w_hu() {
    let a = u32x4::new(2163417444, 940670316, 624242075, 3716350419);
    let b = u16x8::new(10149, 33560, 21613, 61563, 14556, 33558, 30440, 63972);
    let c = u16x8::new(9862, 40610, 42783, 2223, 62194, 15996, 61261, 33667);
    let r = i64x2::new(4627934059328104084, 6765125168025305155);

    assert_eq!(
        r,
        transmute(lsx_vmaddwod_w_hu(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwod_h_bu() {
    let a = u16x8::new(17882, 7508, 14715, 47175, 62895, 51393, 34943, 20707);
    let b = u8x16::new(
        83, 27, 56, 178, 210, 166, 36, 48, 144, 156, 209, 6, 181, 65, 232, 42,
    );
    let c = u8x16::new(
        127, 23, 147, 75, 137, 205, 146, 169, 72, 89, 154, 45, 185, 229, 28, 217,
    );
    let r = i64x2::new(-2884627676759701433, 8394079293504695275);

    assert_eq!(
        r,
        transmute(lsx_vmaddwod_h_bu(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwev_d_wu_w() {
    let a = i64x2::new(-6323015107493705206, -3277448760143472563);
    let b = u32x4::new(2331684563, 1941329953, 2983229925, 1155461882);
    let c = i32x4::new(-1110134113, -106291268, -391880820, 644991581);
    let r = i64x2::new(-8911497681635502825, -4446519349401011063);

    assert_eq!(
        r,
        transmute(lsx_vmaddwev_d_wu_w(
            transmute(a),
            transmute(b),
            transmute(c)
        ))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwev_w_hu_h() {
    let a = i32x4::new(1713941452, 1545069267, -1096163566, -573017556);
    let b = u16x8::new(28055, 23297, 30225, 2761, 48193, 19269, 2518, 51038);
    let c = i16x8::new(-7715, -18819, -4701, -3778, 7207, 5810, -4430, -8060);
    let r = i64x2::new(6025759841279147559, -2509000903003100935);

    assert_eq!(
        r,
        transmute(lsx_vmaddwev_w_hu_h(
            transmute(a),
            transmute(b),
            transmute(c)
        ))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwev_h_bu_b() {
    let a = i16x8::new(27922, 26192, 14273, -18511, -13090, 27036, 4607, 27830);
    let b = u8x16::new(
        85, 234, 241, 30, 218, 135, 230, 175, 34, 217, 231, 43, 159, 81, 198, 89,
    );
    let c = i8x16::new(
        82, -91, 49, -114, 60, -32, -30, 17, 3, 82, -73, -55, -31, -106, -23, -44,
    );
    let r = i64x2::new(-7152443150463563700, 6551891650581220676);

    assert_eq!(
        r,
        transmute(lsx_vmaddwev_h_bu_b(
            transmute(a),
            transmute(b),
            transmute(c)
        ))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwod_d_wu_w() {
    let a = i64x2::new(4995790344325484125, -3678161850757174337);
    let b = u32x4::new(770268311, 2190608617, 3264567056, 3912406971);
    let c = i32x4::new(1039193627, -382136981, 178615845, -2029105420);
    let r = i64x2::new(4158677780872518848, 6829896032850494459);

    assert_eq!(
        r,
        transmute(lsx_vmaddwod_d_wu_w(
            transmute(a),
            transmute(b),
            transmute(c)
        ))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwod_w_hu_h() {
    let a = i32x4::new(-1650648862, 112052630, 369411463, -1789144688);
    let b = u16x8::new(33326, 2589, 54571, 14483, 51494, 10946, 54991, 11715);
    let c = i16x8::new(-13502, 9856, -7830, -1915, 23659, -23776, -29716, 15794);
    let r = i64x2::new(362141702219265378, -6889634254326488121);

    assert_eq!(
        r,
        transmute(lsx_vmaddwod_w_hu_h(
            transmute(a),
            transmute(b),
            transmute(c)
        ))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwod_h_bu_b() {
    let a = i16x8::new(16717, -21485, 6612, -8821, -31304, -13638, -10878, -27550);
    let b = u8x16::new(
        99, 203, 114, 187, 131, 179, 178, 24, 220, 126, 23, 139, 118, 148, 39, 18,
    );
    let c = i8x16::new(
        99, -47, 53, -116, 110, -65, -107, 123, -42, -51, -120, -102, 51, -56, -103, -58,
    );
    let r = i64x2::new(-1651716735493530616, -8048296323958936418);

    assert_eq!(
        r,
        transmute(lsx_vmaddwod_h_bu_b(
            transmute(a),
            transmute(b),
            transmute(c)
        ))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwev_q_d() {
    let a = i64x2::new(-6837031335752177395, -6960992767212208666);
    let b = i64x2::new(-4435069404701670756, -2126315287755608563);
    let c = i64x2::new(-5551390506600609458, -6711686916497928751);
    let r = i64x2::new(-8173734519403794283, -5626296406109360320);

    assert_eq!(
        r,
        transmute(lsx_vmaddwev_q_d(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwod_q_d() {
    let a = i64x2::new(-1677869231369184389, 8708214911109206592);
    let b = i64x2::new(-7813673205639863330, -9004405202552727709);
    let c = i64x2::new(989988865428690976, 7138926957150547746);
    let r = i64x2::new(-1125748635129453663, 5223492036614230927);

    assert_eq!(
        r,
        transmute(lsx_vmaddwod_q_d(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwev_q_du() {
    let a = u64x2::new(17268971871627349752, 17228948998305822956);
    let b = u64x2::new(10411505101371540933, 14258056959108407269);
    let c = u64x2::new(10083084353835617951, 7442290876599468511);
    let r = i64x2::new(4362805751568378451, 4473186691787239539);

    assert_eq!(
        r,
        transmute(lsx_vmaddwev_q_du(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwod_q_du() {
    let a = u64x2::new(14967144687255063091, 6224733010665264496);
    let b = u64x2::new(17625137945884588260, 1535023950244313744);
    let c = u64x2::new(1841326774698258895, 9587959489663720036);
    let r = i64x2::new(1938476888214276723, 7022583698667268618);

    assert_eq!(
        r,
        transmute(lsx_vmaddwod_q_du(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwev_q_du_d() {
    let a = i64x2::new(7413074575332965326, -6131981171876880542);
    let b = u64x2::new(7027881729907986450, 9385132453710384328);
    let c = i64x2::new(6154882990643114022, 8692307970783152636);
    let r = i64x2::new(-8494196038584058246, -3787080112545186901);

    assert_eq!(
        r,
        transmute(lsx_vmaddwev_q_du_d(
            transmute(a),
            transmute(b),
            transmute(c)
        ))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmaddwod_q_du_d() {
    let a = i64x2::new(-3567580028466810679, 82284695558926958);
    let b = u64x2::new(12724355976909764846, 2153966982409398933);
    let c = i64x2::new(-2209580291901273167, -3993952038101553236);
    let r = i64x2::new(-613602630799693851, -384076239737958818);

    assert_eq!(
        r,
        transmute(lsx_vmaddwod_q_du_d(
            transmute(a),
            transmute(b),
            transmute(c)
        ))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vrotr_b() {
    let a = i8x16::new(
        -115, -5, 112, 87, -91, -10, -42, -109, -71, 30, 80, 109, -37, -36, -82, -61,
    );
    let b = i8x16::new(
        98, 80, -27, -51, -44, -43, 28, -49, -47, 12, -100, -113, 35, -85, 9, 23,
    );
    let r = i64x2::new(2841128540244802403, -8694309599374351908);

    assert_eq!(r, transmute(lsx_vrotr_b(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vrotr_h() {
    let a = i16x8::new(29688, -22641, 11287, 9743, 29744, -9683, -24918, 28489);
    let b = i16x8::new(-6485, 1418, 8263, -29872, -6491, 3930, -20621, 32531);
    let r = i64x2::new(2742461657407651598, 3308267577913279393);

    assert_eq!(r, transmute(lsx_vrotr_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vrotr_w() {
    let a = i32x4::new(-232185187, -1057829624, -1428233439, 314333357);
    let b = i32x4::new(1956224189, -1858012941, -1889446514, -2130978943);
    let r = i64x2::new(6458469860191573231, -8548346292466177157);

    assert_eq!(r, transmute(lsx_vrotr_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vrotr_d() {
    let a = i64x2::new(-8694664621869506061, 3293016169868759706);
    let b = i64x2::new(4553458262651691654, -5062393334123159235);
    let r = i64x2::new(-3594618648537251961, 7897385285240526033);

    assert_eq!(r, transmute(lsx_vrotr_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vadd_q() {
    let a = i64x2::new(2423569640801257553, 678073579687698205);
    let b = i64x2::new(114135477458514099, 3481307531297359399);
    let r = i64x2::new(2537705118259771652, 4159381110985057604);

    assert_eq!(r, transmute(lsx_vadd_q(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsub_q() {
    let a = i64x2::new(7892977690518598837, -3112927447911510492);
    let b = i64x2::new(-8526086848853095438, -1323481969747305966);
    let r = i64x2::new(-2027679534337857341, -1789445478164204527);

    assert_eq!(r, transmute(lsx_vsub_q(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vldrepl_b() {
    let a: [i8; 16] = [
        -88, 52, -104, -111, 84, -101, -36, 49, 31, 10, 34, -78, 22, 22, 118, 80,
    ];
    let r = i64x2::new(-6293595036912670552, -6293595036912670552);

    assert_eq!(r, transmute(lsx_vldrepl_b::<0>(a.as_ptr())));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vldrepl_h() {
    let a: [i8; 16] = [
        29, 81, 114, -8, 70, 29, 100, 46, 105, 38, -10, -58, 2, 66, -104, -43,
    ];
    let r = i64x2::new(5844917077753549085, 5844917077753549085);

    assert_eq!(r, transmute(lsx_vldrepl_h::<0>(a.as_ptr())));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vldrepl_w() {
    let a: [i8; 16] = [
        -56, -83, -27, -88, 85, -105, 81, -74, 124, -76, -29, 34, 99, 36, 36, 37,
    ];
    let r = i64x2::new(-6276419428332229176, -6276419428332229176);

    assert_eq!(r, transmute(lsx_vldrepl_w::<0>(a.as_ptr())));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vldrepl_d() {
    let a: [i8; 16] = [
        90, -84, 7, 91, -2, 32, 74, 2, -4, 119, 62, 98, -112, -127, -109, 101,
    ];
    let r = i64x2::new(164980613173455962, 164980613173455962);

    assert_eq!(r, transmute(lsx_vldrepl_d::<0>(a.as_ptr())));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmskgez_b() {
    let a = i8x16::new(
        -121, 102, -85, -2, -103, 100, 119, -46, 35, -16, -66, -43, -61, 79, 40, -43,
    );
    let r = i64x2::new(24930, 0);

    assert_eq!(r, transmute(lsx_vmskgez_b(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vmsknz_b() {
    let a = i8x16::new(
        -25, 93, 124, 56, -119, -93, -123, 118, -27, 16, -22, 58, -59, 69, 63, -66,
    );
    let r = i64x2::new(65535, 0);

    assert_eq!(r, transmute(lsx_vmsknz_b(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vexth_h_b() {
    let a = i8x16::new(
        -86, 119, 29, -97, -55, -30, 39, -102, 85, 73, 20, -12, -94, 53, 30, 114,
    );
    let r = i64x2::new(-3377613816397739, 32088276197572514);

    assert_eq!(r, transmute(lsx_vexth_h_b(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vexth_w_h() {
    let a = i16x8::new(14576, -26514, 14165, -15781, 10106, 1864, 23348, 30478);
    let r = i64x2::new(8005819049850, 130902013270836);

    assert_eq!(r, transmute(lsx_vexth_w_h(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vexth_d_w() {
    let a = i32x4::new(863783254, 799653326, -1122161877, -652869192);
    let r = i64x2::new(-1122161877, -652869192);

    assert_eq!(r, transmute(lsx_vexth_d_w(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vexth_q_d() {
    let a = i64x2::new(2924262436748867523, 1959694872821330818);
    let r = i64x2::new(1959694872821330818, 0);

    assert_eq!(r, transmute(lsx_vexth_q_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vexth_hu_bu() {
    let a = u8x16::new(
        88, 245, 152, 181, 22, 122, 243, 162, 170, 115, 212, 217, 148, 176, 60, 214,
    );
    let r = i64x2::new(61080980486815914, 60235902725652628);

    assert_eq!(r, transmute(lsx_vexth_hu_bu(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vexth_wu_hu() {
    let a = u16x8::new(58875, 18924, 17611, 30197, 33869, 53931, 4693, 53025);
    let r = i64x2::new(231631881274445, 227740640875093);

    assert_eq!(r, transmute(lsx_vexth_wu_hu(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vexth_du_wu() {
    let a = u32x4::new(3499742961, 2840979237, 2082263829, 1096292547);
    let r = i64x2::new(2082263829, 1096292547);

    assert_eq!(r, transmute(lsx_vexth_du_wu(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vexth_qu_du() {
    let a = u64x2::new(14170556367894986991, 14238702840099699193);
    let r = i64x2::new(-4208041233609852423, 0);

    assert_eq!(r, transmute(lsx_vexth_qu_du(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vrotri_b() {
    let a = i8x16::new(
        7, 49, -22, -120, -94, 53, -19, 95, -84, -30, 31, -25, 30, -98, -86, -5,
    );
    let r = i64x2::new(-2919654548887155519, -96080239582005205);

    assert_eq!(r, transmute(lsx_vrotri_b::<2>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vrotri_h() {
    let a = i16x8::new(-14120, -16812, -19570, -990, 24476, -7640, 20329, 8879);
    let r = i64x2::new(-556925602567188047, 4998607264501841720);

    assert_eq!(r, transmute(lsx_vrotri_h::<15>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vrotri_w() {
    let a = i32x4::new(-1760224525, -1644621284, 1835781046, -1487934110);
    let r = i64x2::new(2845787365010917052, -6209343103231659283);

    assert_eq!(r, transmute(lsx_vrotri_w::<2>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vrotri_d() {
    let a = i64x2::new(8884634342417174882, 244175985366916345);
    let r = i64x2::new(-3963790888197019724, 4020656082573561910);

    assert_eq!(r, transmute(lsx_vrotri_d::<52>(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vextl_q_d() {
    let a = i64x2::new(-5110246490938885255, 377414780188285171);
    let r = i64x2::new(-5110246490938885255, -1);

    assert_eq!(r, transmute(lsx_vextl_q_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrlni_b_h() {
    let a = i8x16::new(
        -62, -32, -115, -97, -74, 113, -113, -4, 10, 39, 102, -3, 38, 83, -88, 73,
    );
    let b = i8x16::new(
        115, 89, -35, 113, -13, 93, -90, -127, -73, -66, -71, 19, 37, 76, -89, 116,
    );
    let r = i64x2::new(72339077638193409, 72342367599919619);

    assert_eq!(
        r,
        transmute(lsx_vsrlni_b_h::<14>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrlni_h_w() {
    let a = i16x8::new(4205, -10016, 6553, 16160, 26411, 29470, -20643, 30057);
    let b = i16x8::new(-20939, 15459, 13368, -29800, -25275, -15723, 30837, 7321);
    let r = i64x2::new(1970530997633039, 8162894584676406);

    assert_eq!(
        r,
        transmute(lsx_vsrlni_h_w::<26>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrlni_w_d() {
    let a = i32x4::new(1705975377, 322077350, -1922153156, -661241171);
    let b = i32x4::new(1098943214, -1567917396, 297055649, -1122208150);
    let r = i64x2::new(2133162980935405664, -8022209066041763477);

    assert_eq!(
        r,
        transmute(lsx_vsrlni_w_d::<18>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrlni_d_q() {
    let a = i64x2::new(6325216582707926854, -5129479093920978170);
    let b = i64x2::new(3985485829689892785, 7685789624553197779);
    let r = i64x2::new(7505653930227732, 13005141581824778);

    assert_eq!(
        r,
        transmute(lsx_vsrlni_d_q::<74>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrlrni_b_h() {
    let a = i8x16::new(
        -103, -39, -112, -128, -96, 40, -89, 40, -55, 102, 37, -49, 96, -107, 26, 16,
    );
    let b = i8x16::new(
        -57, 51, 17, 1, 37, 120, -54, 78, -67, 36, 0, -121, -113, 27, -9, 74,
    );
    let r = i64x2::new(3201527803797374159, 4635960605099098726);

    assert_eq!(
        r,
        transmute(lsx_vsrlrni_b_h::<6>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrlrni_h_w() {
    let a = i16x8::new(16435, -5399, -4992, 1377, -27419, -9060, 28877, -12666);
    let b = i16x8::new(30165, -32344, 15225, 17457, -5900, -17127, -30430, 21140);
    let r = i64x2::new(5919251242624655831, 1856453178786227457);

    assert_eq!(
        r,
        transmute(lsx_vsrlrni_h_w::<6>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrlrni_w_d() {
    let a = i32x4::new(-1783593075, -767627057, 522051412, 1497970809);
    let b = i32x4::new(-613709101, 1782777798, -1376237383, -2108949489);
    let r = i64x2::new(8955006813860, 6137508269348);

    assert_eq!(
        r,
        transmute(lsx_vsrlrni_w_d::<52>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrlrni_d_q() {
    let a = i64x2::new(-8390257423140334242, -5915059672723228155);
    let b = i64x2::new(4065462044175592876, 5861150325027293506);
    let r = i64x2::new(42645481, 91180005);

    assert_eq!(
        r,
        transmute(lsx_vsrlrni_d_q::<101>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlni_b_h() {
    let a = i8x16::new(
        -126, 26, 50, 111, 24, 36, -59, -44, -12, 82, 16, -39, 10, 27, -76, -81,
    );
    let b = i8x16::new(
        -72, -74, 3, -16, -50, -40, 17, -39, -88, 33, -11, -74, 27, 104, -56, 35,
    );
    let r = i64x2::new(72907520922224389, 360294575950070528);

    assert_eq!(
        r,
        transmute(lsx_vssrlni_b_h::<13>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlni_h_w() {
    let a = i16x8::new(8928, 556, 327, 11357, -32577, 24481, -16101, -875);
    let b = i16x8::new(12, -2621, -27458, -24262, 23377, 16952, 19498, -31793);
    let r = i64x2::new(74028485831688683, 142145683583401988);

    assert_eq!(
        r,
        transmute(lsx_vssrlni_h_w::<23>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlni_w_d() {
    let a = i32x4::new(1838928968, 1883060425, -990389689, 735664934);
    let b = i32x4::new(-971263991, -98050158, 134746673, -49144118);
    let r = i64x2::new(9223372034707292159, 9223372034707292159);

    assert_eq!(
        r,
        transmute(lsx_vssrlni_w_d::<12>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlni_d_q() {
    let a = i64x2::new(-5470954942766391223, 2164868713336601834);
    let b = i64x2::new(-3507919664178941311, 8800311307152269561);
    let r = i64x2::new(524539429375, 129036230643);

    assert_eq!(
        r,
        transmute(lsx_vssrlni_d_q::<88>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlni_bu_h() {
    let a = u8x16::new(
        42, 80, 7, 61, 49, 172, 110, 186, 30, 201, 214, 72, 201, 231, 144, 223,
    );
    let b = i8x16::new(
        39, 98, -57, 124, 78, 127, 89, 26, 44, 57, 9, -36, -100, -41, 7, 30,
    );
    let r = i64x2::new(1695451225195267, 434318113941815554);

    assert_eq!(
        r,
        transmute(lsx_vssrlni_bu_h::<13>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlni_hu_w() {
    let a = u16x8::new(47562, 12077, 58166, 40959, 47625, 4449, 45497, 47932);
    let b = i16x8::new(25513, -19601, -22702, -15840, 32377, 32023, -4115, 25327);
    let r = i64x2::new(-1, -1);

    assert_eq!(
        r,
        transmute(lsx_vssrlni_hu_w::<9>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlni_wu_d() {
    let a = u32x4::new(3924399037, 1624231459, 1033186938, 4207801648);
    let b = i32x4::new(-343671492, 63408059, -17420952, -742649266);
    let r = i64x2::new(111669149696, 133143986188);

    assert_eq!(
        r,
        transmute(lsx_vssrlni_wu_d::<59>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlni_du_q() {
    let a = u64x2::new(9385373857335523158, 8829548075644432850);
    let b = i64x2::new(1935200102096005901, -4336418136884591685);
    let r = i64x2::new(-1, -1);

    assert_eq!(
        r,
        transmute(lsx_vssrlni_du_q::<6>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlrni_b_h() {
    let a = i8x16::new(
        -118, -53, 124, -32, -8, -106, -30, 125, 80, -118, 111, -49, 2, -54, -109, -63,
    );
    let b = i8x16::new(
        -128, 104, -60, -21, -28, 47, -78, 125, -65, -31, 111, 127, -102, -50, 87, 102,
    );
    let r = i64x2::new(9187201950435737471, 9187201950435737471);

    assert_eq!(
        r,
        transmute(lsx_vssrlrni_b_h::<0>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlrni_h_w() {
    let a = i16x8::new(-6932, -27303, 5931, 1697, 23680, -18344, 21222, 31527);
    let b = i16x8::new(16541, 32147, -26353, -15678, -7913, -31777, 12521, -25215);
    let r = i64x2::new(2814784127631368, 2251851353292809);

    assert_eq!(
        r,
        transmute(lsx_vssrlrni_h_w::<28>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlrni_w_d() {
    let a = i32x4::new(-528492260, 635780412, 2102955910, -106415932);
    let b = i32x4::new(-1062242289, 359654281, 1831754020, 1455206052);
    let r = i64x2::new(9223372034707292159, 9223372034707292159);

    assert_eq!(
        r,
        transmute(lsx_vssrlrni_w_d::<1>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlrni_d_q() {
    let a = i64x2::new(-2050671473765220606, -974956007142498603);
    let b = i64x2::new(4675761647927162976, -5100418369989582579);
    let r = i64x2::new(9223372036854775807, 9223372036854775807);

    assert_eq!(
        r,
        transmute(lsx_vssrlrni_d_q::<60>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlrni_bu_h() {
    let a = u8x16::new(
        100, 79, 212, 163, 219, 225, 100, 84, 1, 173, 146, 41, 33, 251, 175, 18,
    );
    let b = i8x16::new(
        104, -36, 123, 103, -26, -37, -104, -46, 107, -89, 120, 33, 117, -54, 107, 105,
    );
    let r = i64x2::new(217862753078412039, 74310514888869122);

    assert_eq!(
        r,
        transmute(lsx_vssrlrni_bu_h::<13>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlrni_hu_w() {
    let a = u16x8::new(35722, 45502, 51777, 63215, 9369, 33224, 15844, 23578);
    let b = i16x8::new(-18038, 23224, 26314, -15841, 826, -15682, -4109, -24970);
    let r = i64x2::new(22236939778326573, 12948128109625433);

    assert_eq!(
        r,
        transmute(lsx_vssrlrni_hu_w::<25>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlrni_wu_d() {
    let a = u32x4::new(1956924769, 1833875292, 1956412037, 426346371);
    let b = i32x4::new(-1128409795, 198077570, -1649408138, 1665566624);
    let r = i64x2::new(447097136224200392, 114446481822641014);

    assert_eq!(
        r,
        transmute(lsx_vssrlrni_wu_d::<36>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlrni_du_q() {
    let a = u64x2::new(9048079498548224395, 9603999840623079368);
    let b = i64x2::new(-404424089294655868, 5140892317651856748);
    let r = i64x2::new(-1, -1);

    assert_eq!(
        r,
        transmute(lsx_vssrlrni_du_q::<38>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrani_b_h() {
    let a = i8x16::new(
        127, 75, -70, 122, 36, 105, 73, 54, -17, 44, 92, -80, 11, -110, 81, 51,
    );
    let b = i8x16::new(
        -72, 6, 81, -61, -8, -96, 24, 77, 30, -20, 95, -20, 69, -37, -109, 35,
    );
    let r = i64x2::new(2079082344186583605, -7309198813337889445);

    assert_eq!(
        r,
        transmute(lsx_vsrani_b_h::<5>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrani_h_w() {
    let a = i16x8::new(17089, -15383, 6606, 11797, -17230, -236, 24622, 14114);
    let b = i16x8::new(4129, 30226, -29368, -25031, 7609, -18203, 28351, -1400);
    let r = i64x2::new(-8724789849496477438, 2738834860014343212);

    assert_eq!(
        r,
        transmute(lsx_vsrani_h_w::<4>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrani_w_d() {
    let a = i32x4::new(-382819185, 386357255, 35446809, 1387491503);
    let b = i32x4::new(934617213, -1024433792, -516094326, 1363620957);
    let r = i64x2::new(5130829100463783991, -5516717120280852503);

    assert_eq!(
        r,
        transmute(lsx_vsrani_w_d::<24>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrani_d_q() {
    let a = i64x2::new(-6766658862703543347, -8101175034272755526);
    let b = i64x2::new(-6351802365852683233, -7612236351910354649);
    let r = i64x2::new(-58076754393848, -61807060503180);

    assert_eq!(
        r,
        transmute(lsx_vsrani_d_q::<81>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrarni_b_h() {
    let a = i8x16::new(
        -71, 50, -70, -110, 89, 96, -70, 126, 10, 119, -124, -91, -44, -66, -120, -110,
    );
    let b = i8x16::new(
        -118, 101, -58, -7, -118, 69, 75, 88, 75, -76, -41, -37, 13, -46, -84, 68,
    );
    let r = i64x2::new(-7619391791054112335, 5898503720505399127);

    assert_eq!(
        r,
        transmute(lsx_vsrarni_b_h::<3>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrarni_h_w() {
    let a = i16x8::new(-13195, 28211, 7711, -1401, -1145, -27232, 15206, 23526);
    let b = i16x8::new(-21087, 18713, -7401, -30000, 25577, -10794, -28633, -25187);
    let r = i64x2::new(4268193831744344627, -5202735902940537752);

    assert_eq!(
        r,
        transmute(lsx_vsrarni_h_w::<15>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrarni_w_d() {
    let a = i32x4::new(-2004832894, -772030708, -2044339682, -161994376);
    let b = i32x4::new(-314559979, 1401503238, -738119523, -2036313194);
    let r = i64x2::new(-64424509430, -6);

    assert_eq!(
        r,
        transmute(lsx_vsrarni_w_d::<59>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vsrarni_d_q() {
    let a = i64x2::new(2532701208156415278, 7815982649469220899);
    let b = i64x2::new(-202407401251467620, 284380589150850504);
    let r = i64x2::new(-202407401251467620, 2532701208156415278);

    assert_eq!(
        r,
        transmute(lsx_vsrarni_d_q::<0>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrani_b_h() {
    let a = i8x16::new(
        -50, 30, 4, -123, 102, 17, -127, 79, -3, 54, -91, 77, -81, -74, -32, 6,
    );
    let b = i8x16::new(
        -125, 114, -41, -31, 70, 17, -109, 98, -43, -79, -24, -39, -79, 49, -43, 61,
    );
    let r = i64x2::new(9187203054242332799, 9187483425412448383);

    assert_eq!(
        r,
        transmute(lsx_vssrani_b_h::<0>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrani_h_w() {
    let a = i16x8::new(-13653, 21802, 26851, -30910, -21293, -13050, -24174, 29805);
    let b = i16x8::new(9604, -27726, -18692, 147, 23503, 3941, -18536, -25864);
    let r = i64x2::new(-1970324836909063, 2251786928259077);

    assert_eq!(
        r,
        transmute(lsx_vssrani_h_w::<28>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrani_w_d() {
    let a = i32x4::new(640738652, 568129780, 2099035547, 1750495014);
    let b = i32x4::new(2090153020, 2002243310, 567374078, -1386845950);
    let r = i64x2::new(-45445048943701, 57359288242414);

    assert_eq!(
        r,
        transmute(lsx_vssrani_w_d::<49>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrani_d_q() {
    let a = i64x2::new(8313689526826187568, -7067970090029512662);
    let b = i64x2::new(-7547166008384655380, 9056943104343751836);
    let r = i64x2::new(138197984380245, -107848664703820);

    assert_eq!(
        r,
        transmute(lsx_vssrani_d_q::<80>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrani_bu_h() {
    let a = u8x16::new(
        110, 23, 112, 128, 94, 127, 141, 246, 144, 229, 149, 191, 73, 211, 119, 89,
    );
    let b = i8x16::new(
        9, -116, 68, -122, 13, -17, -90, 29, -22, -126, 50, 2, -50, -121, 124, -18,
    );
    let r = i64x2::new(0, 72057594037993472);

    assert_eq!(
        r,
        transmute(lsx_vssrani_bu_h::<14>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrani_hu_w() {
    let a = u16x8::new(23583, 19333, 39698, 13735, 15385, 8819, 61012, 57430);
    let b = i16x8::new(-18676, -5045, 14040, 25346, -27192, -27172, 13333, 12330);
    let r = i64x2::new(27021597777199104, 292064788631);

    assert_eq!(
        r,
        transmute(lsx_vssrani_hu_w::<23>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrani_wu_d() {
    let a = u32x4::new(3826341651, 1946901217, 3504547080, 2702234829);
    let b = i32x4::new(1013240156, -1783678601, -91667235, 485058283);
    let r = i64x2::new(-4294967296, 4294967295);

    assert_eq!(
        r,
        transmute(lsx_vssrani_wu_d::<13>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrani_du_q() {
    let a = u64x2::new(16452622598975149813, 15788367695672970142);
    let b = i64x2::new(3271075037846423078, -4777595873776840194);
    let r = i64x2::new(0, 0);

    assert_eq!(
        r,
        transmute(lsx_vssrani_du_q::<33>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrarni_b_h() {
    let a = i8x16::new(
        -76, 3, 89, 123, 98, -91, 87, 101, 75, 77, -114, 117, -78, 10, -64, 13,
    );
    let b = i8x16::new(
        125, 49, 97, -128, -38, 61, 29, 1, -108, 54, 28, -65, -22, -3, 71, -12,
    );
    let r = i64x2::new(-9187201955687071617, 9187201950435803007);

    assert_eq!(
        r,
        transmute(lsx_vssrarni_b_h::<2>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrarni_h_w() {
    let a = i16x8::new(-5012, 11989, 5954, -22500, 4485, 31359, 28715, -16160);
    let b = i16x8::new(29828, -15046, 20055, -7703, 18306, -411, -15337, 30957);
    let r = i64x2::new(1125904201809918, -562928478781439);

    assert_eq!(
        r,
        transmute(lsx_vssrarni_h_w::<29>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrarni_w_d() {
    let a = i32x4::new(830116125, -782674123, 1854407155, 1495209920);
    let b = i32x4::new(2038928041, -944152498, 984207668, -1562095866);
    let r = i64x2::new(-9223372034707292160, 9223372034707292160);

    assert_eq!(
        r,
        transmute(lsx_vssrarni_w_d::<18>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrarni_d_q() {
    let a = i64x2::new(6798655171089504447, 7326163030789656624);
    let b = i64x2::new(-2977477884402038599, -1140443471327573805);
    let r = i64x2::new(-17819429239493341, 114471297356088385);

    assert_eq!(
        r,
        transmute(lsx_vssrarni_d_q::<70>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrarni_bu_h() {
    let a = u8x16::new(
        75, 193, 237, 8, 33, 177, 31, 133, 119, 169, 163, 98, 159, 36, 131, 221,
    );
    let b = i8x16::new(
        85, 84, -17, -84, 37, -124, -96, -30, -113, 114, -49, -7, 93, -3, -69, 124,
    );
    let r = i64x2::new(144115196665790465, 283673999966208);

    assert_eq!(
        r,
        transmute(lsx_vssrarni_bu_h::<14>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrarni_hu_w() {
    let a = u16x8::new(24614, 57570, 38427, 46010, 4180, 57175, 13134, 32047);
    let b = i16x8::new(20333, -10949, -20123, -1525, 14594, -30628, -30604, -29092);
    let r = i64x2::new(0, -281474976710656);

    assert_eq!(
        r,
        transmute(lsx_vssrarni_hu_w::<13>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrarni_wu_d() {
    let a = u32x4::new(1854465345, 2301618375, 1724286997, 3204532825);
    let b = i32x4::new(-1176670423, -1482282410, 777914585, 87761646);
    let r = i64x2::new(-4294967296, 0);

    assert_eq!(
        r,
        transmute(lsx_vssrarni_wu_d::<15>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrarni_du_q() {
    let a = u64x2::new(5657125151084901446, 434040259538460448);
    let b = i64x2::new(4567159404230772553, -10612253426094316);
    let r = i64x2::new(0, 0);

    assert_eq!(
        r,
        transmute(lsx_vssrarni_du_q::<126>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vpermi_w() {
    let a = i32x4::new(213291370, -674346961, -1480878002, -1600622413);
    let b = i32x4::new(-1309240039, 1335257352, 852153543, 1125109318);
    let r = i64x2::new(4832307726087017671, -6360322584335202257);

    assert_eq!(
        r,
        transmute(lsx_vpermi_w::<158>(transmute(a), transmute(b)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vld() {
    let a: [i8; 16] = [
        127, 127, 77, 66, 64, 25, -50, -34, 2, -7, 107, -87, 45, -88, -51, 41,
    ];
    let r = i64x2::new(-2391946588306178177, 3012248639850150146);

    assert_eq!(r, transmute(lsx_vld::<0>(a.as_ptr())));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vst() {
    let a = i8x16::new(
        -27, -57, 84, 27, -46, -85, -92, 57, 15, -67, -44, -89, -88, 84, 22, -29,
    );
    let mut o: [i8; 16] = [
        -9, 24, -11, -95, -10, 78, 41, -118, 91, -113, 107, 77, -50, 113, -22, 27,
    ];
    let r = i64x2::new(4153633675232462821, -2083384694265299697);

    lsx_vst::<0>(transmute(a), o.as_mut_ptr());
    assert_eq!(r, transmute(o));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlrn_b_h() {
    let a = i16x8::new(-6731, 13740, 8488, -2854, -3028, 6907, -57, 5317);
    let b = i16x8::new(17437, 9775, -20467, -31838, 5913, 4238, -7458, 2822);
    let r = i64x2::new(5981906731171643399, 0);

    assert_eq!(r, transmute(lsx_vssrlrn_b_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlrn_h_w() {
    let a = i32x4::new(1684402804, 1385352714, 1360229118, 928996904);
    let b = i32x4::new(-2116426818, 1641049288, 712377342, -1572394121);
    let r = i64x2::new(31243728857268226, 0);

    assert_eq!(r, transmute(lsx_vssrlrn_h_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrlrn_w_d() {
    let a = i64x2::new(-6889047968033387497, -1417681658907465534);
    let b = i64x2::new(-3890929847852895653, -7819301294522132056);
    let r = i64x2::new(66519777023098879, 0);

    assert_eq!(r, transmute(lsx_vssrlrn_w_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrln_b_h() {
    let a = i16x8::new(6474, 27187, -10340, 1859, 23966, -18880, 3680, 9203);
    let b = i16x8::new(-14062, -29610, -24609, -8884, -1818, 32133, 29934, -6498);
    let r = i64x2::new(140183437672319, 0);

    assert_eq!(r, transmute(lsx_vssrln_b_h(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrln_h_w() {
    let a = i32x4::new(-476821436, -709684595, 1401465952, -1429729676);
    let b = i32x4::new(-1437891045, 1546371535, -1800954476, -1892390372);
    let r = i64x2::new(2820489990832156, 0);

    assert_eq!(r, transmute(lsx_vssrln_h_w(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vssrln_w_d() {
    let a = i64x2::new(2563829598589943649, 1915912925013067420);
    let b = i64x2::new(2034490755997557661, -3470252066162700534);
    let r = i64x2::new(9223372034707292159, 0);

    assert_eq!(r, transmute(lsx_vssrln_w_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vorn_v() {
    let a = i8x16::new(
        -104, -56, -109, -5, -124, 58, 19, -45, -64, 70, 0, 60, -67, -86, -77, -47,
    );
    let b = i8x16::new(
        18, 99, -128, 74, -16, -127, 71, 94, -99, -119, 16, 43, 121, 77, -57, -24,
    );
    let r = i64x2::new(-883973744907789059, -2901520201165080862);

    assert_eq!(r, transmute(lsx_vorn_v(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vldi() {
    let r = i64x2::new(-404, -404);

    assert_eq!(r, transmute(lsx_vldi::<3692>()));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vshuf_b() {
    let a = i8x16::new(
        115, -20, -59, -22, 43, -85, -79, 110, -79, -97, 14, -11, 5, -43, 17, -16,
    );
    let b = i8x16::new(
        -49, -101, -67, -10, -11, 76, -1, -74, 10, 110, 27, -53, 105, 34, 28, 98,
    );
    let c = i8x16::new(3, 10, 3, 20, 23, 29, 7, 23, 3, 3, 4, 15, 3, 10, 21, 27);
    let r = i64x2::new(7977798459094080502, -744470568363493642);

    assert_eq!(
        r,
        transmute(lsx_vshuf_b(transmute(a), transmute(b), transmute(c)))
    );
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vldx() {
    let a: [i8; 16] = [
        -102, -39, 3, 31, 58, -5, 78, 11, -96, -111, 11, 114, 103, -3, -86, 37,
    ];
    let r = i64x2::new(814864809647659418, 2714260346180964768);

    assert_eq!(r, transmute(lsx_vldx(a.as_ptr(), 0)));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vstx() {
    let a = i8x16::new(
        113, -106, 22, -4, 54, 56, 70, -21, -30, 0, -25, -98, 56, -46, -51, 99,
    );
    let mut o: [i8; 16] = [
        -60, -30, -98, 12, 90, 96, 120, -102, -124, 54, -91, -24, 126, -80, 121, -29,
    ];
    let r = i64x2::new(-1493444417618012559, 7191635320606490850);

    lsx_vstx(transmute(a), o.as_mut_ptr(), 0);
    assert_eq!(r, transmute(o));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vextl_qu_du() {
    let a = u64x2::new(14708598110732796778, 2132245682694336458);
    let r = i64x2::new(-3738145962976754838, 0);

    assert_eq!(r, transmute(lsx_vextl_qu_du(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_bnz_b() {
    let a = u8x16::new(
        84, 211, 197, 223, 221, 228, 88, 147, 165, 38, 137, 91, 54, 252, 130, 198,
    );
    let r: i32 = 1;

    assert_eq!(r, transmute(lsx_bnz_b(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_bnz_d() {
    let a = u64x2::new(2935166648440262530, 9853932033129373129);
    let r: i32 = 1;

    assert_eq!(r, transmute(lsx_bnz_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_bnz_h() {
    let a = u16x8::new(55695, 60003, 59560, 35123, 25693, 41352, 61626, 42007);
    let r: i32 = 1;

    assert_eq!(r, transmute(lsx_bnz_h(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_bnz_v() {
    let a = u8x16::new(
        97, 136, 236, 21, 16, 18, 39, 247, 250, 7, 67, 251, 83, 240, 242, 151,
    );
    let r: i32 = 1;

    assert_eq!(r, transmute(lsx_bnz_v(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_bnz_w() {
    let a = u32x4::new(1172712391, 4211490091, 1954893853, 1606462106);
    let r: i32 = 1;

    assert_eq!(r, transmute(lsx_bnz_w(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_bz_b() {
    let a = u8x16::new(
        15, 239, 121, 77, 200, 213, 232, 133, 158, 104, 98, 165, 77, 238, 68, 228,
    );
    let r: i32 = 0;

    assert_eq!(r, transmute(lsx_bz_b(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_bz_d() {
    let a = u64x2::new(6051854163594201075, 9957257179760945130);
    let r: i32 = 0;

    assert_eq!(r, transmute(lsx_bz_d(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_bz_h() {
    let a = u16x8::new(19470, 29377, 53886, 60432, 20799, 41755, 54479, 52192);
    let r: i32 = 0;

    assert_eq!(r, transmute(lsx_bz_h(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_bz_v() {
    let a = u8x16::new(
        205, 20, 220, 220, 212, 207, 232, 167, 86, 81, 26, 68, 30, 112, 186, 234,
    );
    let r: i32 = 0;

    assert_eq!(r, transmute(lsx_bz_v(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_bz_w() {
    let a = u32x4::new(840335855, 1404686204, 628335401, 1171808080);
    let r: i32 = 0;

    assert_eq!(r, transmute(lsx_bz_w(transmute(a))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_caf_d() {
    let a = u64x2::new(4603762778598497410, 4600578720825355240);
    let b = u64x2::new(4594845432849836188, 4605165420863530034);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfcmp_caf_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_caf_s() {
    let a = u32x4::new(1057450480, 1041717868, 1063383650, 1052061330);
    let b = u32x4::new(1058412800, 1058762495, 1028487696, 1027290752);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfcmp_caf_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_ceq_d() {
    let a = u64x2::new(4605168921160906654, 4594290648143726556);
    let b = u64x2::new(4605937250150464526, 4596769502461699132);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfcmp_ceq_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_ceq_s() {
    let a = u32x4::new(1022481472, 1054281004, 1061611781, 1063964926);
    let b = u32x4::new(1057471620, 1064008655, 1062698831, 1064822930);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfcmp_ceq_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_cle_d() {
    let a = u64x2::new(4594614911097184960, 4595883006410794928);
    let b = u64x2::new(4596931282408842596, 4592481315209481584);
    let r = i64x2::new(-1, 0);

    assert_eq!(r, transmute(lsx_vfcmp_cle_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_cle_s() {
    let a = u32x4::new(1056795676, 1033595408, 1059655467, 1052539946);
    let b = u32x4::new(1021993344, 1043028808, 1064182329, 1054794412);
    let r = i64x2::new(-4294967296, -1);

    assert_eq!(r, transmute(lsx_vfcmp_cle_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_clt_d() {
    let a = u64x2::new(4600913855630793750, 4577092243808815872);
    let b = u64x2::new(4603056125735978454, 4595932368389116476);
    let r = i64x2::new(-1, -1);

    assert_eq!(r, transmute(lsx_vfcmp_clt_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_clt_s() {
    let a = u32x4::new(1056969130, 1052243316, 1061133360, 1024378560);
    let b = u32x4::new(1040327468, 1040072248, 1063314103, 1061361061);
    let r = i64x2::new(0, -1);

    assert_eq!(r, transmute(lsx_vfcmp_clt_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_cne_d() {
    let a = u64x2::new(4600626466477018126, 4598733447126827764);
    let b = u64x2::new(4602354759349431170, 4598595124838935466);
    let r = i64x2::new(-1, -1);

    assert_eq!(r, transmute(lsx_vfcmp_cne_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_cne_s() {
    let a = u32x4::new(1063546111, 1053175192, 1063179686, 1052800226);
    let b = u32x4::new(1063262940, 1058010357, 1052721962, 1061295988);
    let r = i64x2::new(-1, -1);

    assert_eq!(r, transmute(lsx_vfcmp_cne_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_cor_d() {
    let a = u64x2::new(4607018705522720912, 4606390725849766769);
    let b = u64x2::new(4606863361114437050, 4600753700959452152);
    let r = i64x2::new(-1, -1);

    assert_eq!(r, transmute(lsx_vfcmp_cor_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_cor_s() {
    let a = u32x4::new(993114880, 1063738833, 1020144864, 1055277186);
    let b = u32x4::new(1053615382, 1065255138, 1051565294, 1041776832);
    let r = i64x2::new(-1, -1);

    assert_eq!(r, transmute(lsx_vfcmp_cor_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_cueq_d() {
    let a = u64x2::new(4589986692503775384, 4604350239975880608);
    let b = u64x2::new(4603317345052528721, 4586734343919602352);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfcmp_cueq_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_cueq_s() {
    let a = u32x4::new(1049781896, 1063241920, 1063535787, 1062764831);
    let b = u32x4::new(1057082822, 1059761998, 1052599998, 1054369118);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfcmp_cueq_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_cule_d() {
    let a = u64x2::new(4600113342137410192, 4586591372067099760);
    let b = u64x2::new(4604253448175093958, 4599648167588382448);
    let r = i64x2::new(-1, -1);

    assert_eq!(r, transmute(lsx_vfcmp_cule_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_cule_s() {
    let a = u32x4::new(1059878844, 1040845348, 1060450143, 1061437832);
    let b = u32x4::new(1051100696, 1062219104, 1064568294, 1032521352);
    let r = i64x2::new(-4294967296, 4294967295);

    assert_eq!(r, transmute(lsx_vfcmp_cule_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_cult_d() {
    let a = u64x2::new(4604916546627232568, 4599229615347667200);
    let b = u64x2::new(4602944708025910986, 4606429728449082215);
    let r = i64x2::new(0, -1);

    assert_eq!(r, transmute(lsx_vfcmp_cult_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_cult_s() {
    let a = u32x4::new(1061581945, 1058257026, 1059733857, 1064954284);
    let b = u32x4::new(1030808384, 1044268840, 1050761328, 1037308928);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfcmp_cult_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_cun_d() {
    let a = u64x2::new(4603128178250554600, 4601297724275716756);
    let b = u64x2::new(4599145506416791474, 4602762942707610466);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfcmp_cun_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_cune_d() {
    let a = u64x2::new(4603159382334199523, 4603135754641654385);
    let b = u64x2::new(4602895209237804084, 4598685577984089858);
    let r = i64x2::new(-1, -1);

    assert_eq!(r, transmute(lsx_vfcmp_cune_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_cune_s() {
    let a = u32x4::new(1059907972, 1059391341, 1025259296, 1050646758);
    let b = u32x4::new(1049955876, 1032474200, 1023410112, 1050347912);
    let r = i64x2::new(-1, -1);

    assert_eq!(r, transmute(lsx_vfcmp_cune_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_cun_s() {
    let a = u32x4::new(1054871898, 1059065315, 1037157736, 1056161416);
    let b = u32x4::new(1053288920, 1059911123, 1058695573, 1062913175);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfcmp_cun_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_saf_d() {
    let a = u64x2::new(4585010456558902064, 4598376734249785852);
    let b = u64x2::new(4589118818065931376, 4603302333347826011);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfcmp_saf_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_saf_s() {
    let a = u32x4::new(1039827304, 1062400770, 1052695470, 1056530338);
    let b = u32x4::new(1044756936, 1054667546, 1059141760, 1062203553);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfcmp_saf_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_seq_d() {
    let a = u64x2::new(4604896813051509737, 4596873540510119820);
    let b = u64x2::new(4594167956310606988, 4596272126122589228);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfcmp_seq_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_seq_s() {
    let a = u32x4::new(1060477925, 1048954814, 1059933669, 1053469148);
    let b = u32x4::new(1057231588, 1051495460, 1057998997, 1049117328);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfcmp_seq_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_sle_d() {
    let a = u64x2::new(4605211142905317821, 4601961488287203912);
    let b = u64x2::new(4603919005855163252, 4594682846653946884);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfcmp_sle_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_sle_s() {
    let a = u32x4::new(1053671520, 1055456634, 1063294891, 1059790187);
    let b = u32x4::new(1045989468, 1052518900, 1046184640, 1032417352);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfcmp_sle_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_slt_d() {
    let a = u64x2::new(4601902750800060998, 4605236132294100877);
    let b = u64x2::new(4600564867142526828, 4585131890265864544);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfcmp_slt_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_slt_s() {
    let a = u32x4::new(1054326748, 1059604229, 1060884737, 1022762624);
    let b = u32x4::new(1063435026, 1062439603, 1060665555, 1059252630);
    let r = i64x2::new(-1, -4294967296);

    assert_eq!(r, transmute(lsx_vfcmp_slt_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_sne_d() {
    let a = u64x2::new(4606672121388401433, 4604186491240191582);
    let b = u64x2::new(4606789952952688555, 4605380358192261377);
    let r = i64x2::new(-1, -1);

    assert_eq!(r, transmute(lsx_vfcmp_sne_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_sne_s() {
    let a = u32x4::new(1062253602, 1053568536, 1056615768, 1055754482);
    let b = u32x4::new(1055803760, 1063372602, 1062608900, 1054634370);
    let r = i64x2::new(-1, -1);

    assert_eq!(r, transmute(lsx_vfcmp_sne_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_sor_d() {
    let a = u64x2::new(4595713406002022116, 4604653971232015460);
    let b = u64x2::new(4606380175568635560, 4602092067387067462);
    let r = i64x2::new(-1, -1);

    assert_eq!(r, transmute(lsx_vfcmp_sor_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_sor_s() {
    let a = u32x4::new(1058728243, 1059025743, 1012810944, 1057593472);
    let b = u32x4::new(1064534350, 1035771168, 1059142426, 1034677600);
    let r = i64x2::new(-1, -1);

    assert_eq!(r, transmute(lsx_vfcmp_sor_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_sueq_d() {
    let a = u64x2::new(4605322679929877488, 4603091890812380784);
    let b = u64x2::new(4602917609947054533, 4605983209212177197);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfcmp_sueq_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_sueq_s() {
    let a = u32x4::new(1058057744, 1049762394, 1044222368, 1050250466);
    let b = u32x4::new(1064871165, 1059796257, 1055456352, 1058662692);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfcmp_sueq_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_sule_d() {
    let a = u64x2::new(4606210463692472427, 4576137083667840000);
    let b = u64x2::new(4594044173266256632, 4601549551994738386);
    let r = i64x2::new(0, -1);

    assert_eq!(r, transmute(lsx_vfcmp_sule_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_sule_s() {
    let a = u32x4::new(1054399614, 1064056006, 1040844632, 1022950656);
    let b = u32x4::new(1061061244, 1051874412, 1041025316, 1056018690);
    let r = i64x2::new(4294967295, -1);

    assert_eq!(r, transmute(lsx_vfcmp_sule_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_sult_d() {
    let a = u64x2::new(4593772214968107560, 4602360976974434088);
    let b = u64x2::new(4603848042095479627, 4605032971316970060);
    let r = i64x2::new(-1, -1);

    assert_eq!(r, transmute(lsx_vfcmp_sult_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_sult_s() {
    let a = u32x4::new(1055857986, 1049674182, 1050153588, 1054289234);
    let b = u32x4::new(1053631630, 1064026599, 1058029398, 1041182304);
    let r = i64x2::new(-4294967296, 4294967295);

    assert_eq!(r, transmute(lsx_vfcmp_sult_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_sun_d() {
    let a = u64x2::new(4600661687369290390, 4583739657744995904);
    let b = u64x2::new(4560681020073292800, 4604624347352815433);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfcmp_sun_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_sune_d() {
    let a = u64x2::new(4600101879341653256, 4602392889952410448);
    let b = u64x2::new(4593947987798339484, 4603656097008761637);
    let r = i64x2::new(-1, -1);

    assert_eq!(r, transmute(lsx_vfcmp_sune_d(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_sune_s() {
    let a = u32x4::new(1058419193, 1062297121, 1026375712, 1061355356);
    let b = u32x4::new(1049327168, 1034635272, 1042258196, 1062844003);
    let r = i64x2::new(-1, -1);

    assert_eq!(r, transmute(lsx_vfcmp_sune_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vfcmp_sun_s() {
    let a = u32x4::new(1044637928, 1061035459, 1051032716, 1050118110);
    let b = u32x4::new(1057442863, 1064573466, 1058086753, 1015993248);
    let r = i64x2::new(0, 0);

    assert_eq!(r, transmute(lsx_vfcmp_sun_s(transmute(a), transmute(b))));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vrepli_b() {
    let r = i64x2::new(4340410370284600380, 4340410370284600380);

    assert_eq!(r, transmute(lsx_vrepli_b::<-452>()));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vrepli_d() {
    let r = i64x2::new(-330, -330);

    assert_eq!(r, transmute(lsx_vrepli_d::<-330>()));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vrepli_h() {
    let r = i64x2::new(39125618772344971, 39125618772344971);

    assert_eq!(r, transmute(lsx_vrepli_h::<139>()));
}

#[simd_test(enable = "lsx")]
unsafe fn test_lsx_vrepli_w() {
    let r = i64x2::new(-468151435374, -468151435374);

    assert_eq!(r, transmute(lsx_vrepli_w::<-110>()));
}
