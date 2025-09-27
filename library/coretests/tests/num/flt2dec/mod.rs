use core::num::flt2dec::decoder::{Decoded64, decode_f16, decode_f32, decode_f64};
use core::num::flt2dec::strategy::{dragon, grisu};
use std::mem::MaybeUninit;
use std::str;

use crate::num::{ldexp_f32, ldexp_f64};

mod estimator;
mod strategy {
    mod dragon;
    mod grisu;
}
mod random;

fn try_short(label: &str, dec: &Decoded64, want_digits: &str, want_exp: isize) {
    let want = (want_digits, want_exp);
    let mut buf = [MaybeUninit::<u8>::uninit(); 17];

    let got = dragon::format_short(dec, &mut buf);
    assert_eq!(got, want, "format short of {label} with dragon");

    if let Some(got) = grisu::format_short(dec, &mut buf) {
        assert_eq!(got, want, "format short of {label} with grisu");
    }
}

fn try_fixed_unlimited(label: &str, dec: &Decoded64, want_digits: &str, want_exp: isize) {
    let want = (want_digits, want_exp);
    let mut buf = [MaybeUninit::<u8>::uninit(); 1024];
    let sized_buf = &mut buf[..want_digits.len()];

    let got = dragon::format_fixed(dec, sized_buf, isize::MIN);
    assert_eq!(got, want, "format fixed of {label} with dragon");

    if let Some(got) = grisu::format_fixed(dec, sized_buf, isize::MIN) {
        assert_eq!(got, want, "format fixed of {label} with grisu");
    }
}

fn try_fixed(
    label: &str,
    dec: &Decoded64,
    buf: &mut [std::mem::MaybeUninit<u8>; 1024],
    limit: isize,
    want_digits: &str,
    want_exp: isize,
) {
    let want = (want_digits, want_exp);

    let got = dragon::format_fixed(dec, buf, limit);
    assert_eq!(got, want, "format fixed of {label}, limit {limit}, with dragon");

    if let Some(got) = grisu::format_fixed(dec, buf, limit) {
        assert_eq!(got, want, "format fixed of {label}, limit {limit}, with grisu");
    }
}

fn check_fixed(label: &str, dec: &Decoded64, want_digits: &str, want_exp: isize) {
    try_fixed_unlimited(label, &dec, want_digits, want_exp);
    let mut buf = [MaybeUninit::new(b'_'); 1024];
    try_fixed(label, &dec, &mut buf, want_exp - want_digits.len() as isize, want_digits, want_exp);

    // check exact rounding for zero- and negative-width cases
    let start = if want_digits.chars().next().unwrap() > '5' {
        try_fixed(label, &dec, &mut buf, want_exp, "1", want_exp + 1);
        1
    } else {
        0
    };
    for i in start..-10 {
        try_fixed(label, &dec, &mut buf, want_exp - i, "", want_exp);
    }
}

macro_rules! check_types {
    ($($T:ident)*) => {
        $(

        macro_rules! ${concat(check_short_, $T)} {
            ($v:expr => $want_digits:expr, $want_exp:expr) => {
                assert!(($v as $T).is_finite());
                let dec = ${concat(decode_, $T)}($v);
                try_short(stringify!($v), &dec, $want_digits, $want_exp);
            };
        }

        macro_rules! ${concat(check_fixed_, $T)} {
            ($v:expr => $want_digits:expr, $want_exp:expr) => {
                assert!(($v as $T).is_finite());
                let dec = ${concat(decode_, $T)}($v);
                check_fixed(stringify!($v), &dec, $want_digits, $want_exp);
            };
        }

        #[allow(unused_macros)]
        macro_rules! ${concat(check_fixed_infz_, $T)} {
            ($v:expr => $want_digits:expr, $want_exp:expr) => {
                assert!(($v as $T).is_finite());
                let dec = ${concat(decode_, $T)}($v);
                let digits: &str = $want_digits;
                check_fixed(stringify!($v), &dec, digits, $want_exp);

                let label = concat!(stringify!($v), " with zero trail");
                let mut buf = [MaybeUninit::<u8>::uninit(); 1024];

                // Verify "infinite" zero digits.
                let mut zero_trail = [b'0'; 1024];
                zero_trail[..digits.len()].copy_from_slice(digits.as_bytes());
                let with_zeroes = str::from_utf8(&zero_trail).unwrap();
                for ndig in (digits.len() + 1)..zero_trail.len() {
                    try_fixed(label, &dec, &mut buf, $want_exp - ndig as isize, &with_zeroes[..ndig], $want_exp);
                }
            };
        }

        #[allow(unused_macros)]
        macro_rules! ${concat(check_fixed_one_, $T)} {
            ($x:expr, $e:expr => $want_digits:expr, $want_exp:expr) => {
                let label = concat!(stringify!($x), " * 2^", stringify!($e), " as ", stringify!($T));
                let x: $T = $x;
                let e: isize = $e;
                let dec = ${concat(decode_, $T)}(x * (e as $T).exp2());

                try_fixed_unlimited(label, &dec, $want_digits, $want_exp);

                let limit = $want_exp - $want_digits.len() as isize;
                let mut buf = [MaybeUninit::new(b'_'); 1024];
                try_fixed(label, &dec, &mut buf, limit, $want_digits, $want_exp);
            };
        }


        )*
    };
}

check_types! { f16 f32 f64 }

// in the following comments, three numbers are spaced by 1 ulp apart,
// and the second one is being formatted.
//
// some tests are derived from [1].
//
// [1] Vern Paxson, A Program for Testing IEEE Decimal-Binary Conversion
//     ftp://ftp.ee.lbl.gov/testbase-report.ps.Z
//  or https://www.icir.org/vern/papers/testbase-report.pdf

#[test]
#[cfg(target_has_reliable_f16)]
pub fn f16_short_sanity_test() {
    // 0.0999145507813
    // 0.0999755859375
    // 0.100036621094
    check_short_f16!(0.1 => "1", 0);

    // 0.3330078125
    // 0.333251953125 (1/3 in the default rounding)
    // 0.33349609375
    check_short_f16!(1.0/3.0 => "3333", 0);

    // 10^1 * 0.3138671875
    // 10^1 * 0.3140625
    // 10^1 * 0.3142578125
    check_short_f16!(3.14 => "314", 1);

    // 10^18 * 0.31415916243714048
    // 10^18 * 0.314159196796878848
    // 10^18 * 0.314159231156617216
    check_short_f16!(3.1415e4 => "3141", 5);

    // regression test for decoders
    // 10^2 * 0.31984375
    // 10^2 * 0.32
    // 10^2 * 0.3203125
    check_short_f16!(crate::num::ldexp_f16(1.0, 5) => "32", 2);

    // 10^5 * 0.65472
    // 10^5 * 0.65504
    // 10^5 * 0.65536
    check_short_f16!(f16::MAX => "655", 5);

    // 10^-4 * 0.60975551605224609375
    // 10^-4 * 0.6103515625
    // 10^-4 * 0.61094760894775390625
    check_short_f16!(f16::MIN_POSITIVE => "6104", -4);

    // 10^-9 * 0
    // 10^-9 * 0.59604644775390625
    // 10^-8 * 0.11920928955078125
    check_short_f16!(crate::num::ldexp_f16(1.0, -24) => "6", -7);
}

#[test]
#[cfg(target_has_reliable_f16)]
#[cfg_attr(miri, ignore)] // Miri is too slow
pub fn f16_exact_sanity_test() {
    check_fixed_infz_f16!(0.1               => "999755859375", -1);
    check_fixed_infz_f16!(0.5               => "5", 0);
    check_fixed_infz_f16!(1.0/3.0           => "333251953125", 0);
    check_fixed_infz_f16!(3.141             => "3140625", 1);
    check_fixed_infz_f16!(3.141e4           => "31408", 5);
    check_fixed_infz_f16!(f16::MAX          => "65504", 5);
    check_fixed_infz_f16!(f16::MIN_POSITIVE => "6103515625", -4);
    check_fixed_f16!(crate::num::ldexp_f16(1.0, -24) => "59604644775390625", -7);

    // FIXME(f16_f128): these should gain the check_fixed_one tests like `f32` and `f64` have,
    // but these values are not easy to generate. The algorithm from the Paxon paper [1] needs
    // to be adapted to binary16.
}

#[test]
pub fn f32_short_sanity_test() {
    // 0.0999999940395355224609375
    // 0.100000001490116119384765625
    // 0.10000000894069671630859375
    check_short_f32!(0.1 => "1", 0);

    // 0.333333313465118408203125
    // 0.3333333432674407958984375 (1/3 in the default rounding)
    // 0.33333337306976318359375
    check_short_f32!(1.0/3.0 => "33333334", 0);

    // 10^1 * 0.31415917873382568359375
    // 10^1 * 0.31415920257568359375
    // 10^1 * 0.31415922641754150390625
    check_short_f32!(3.141592 => "3141592", 1);

    // 10^18 * 0.31415916243714048
    // 10^18 * 0.314159196796878848
    // 10^18 * 0.314159231156617216
    check_short_f32!(3.141592e17 => "3141592", 18);

    // regression test for decoders
    // 10^8 * 0.3355443
    // 10^8 * 0.33554432
    // 10^8 * 0.33554436
    check_short_f32!(ldexp_f32(1.0, 25) => "33554432", 8);

    // 10^39 * 0.340282326356119256160033759537265639424
    // 10^39 * 0.34028234663852885981170418348451692544
    // 10^39 * 0.340282366920938463463374607431768211456
    check_short_f32!(f32::MAX => "34028235", 39);

    // 10^-37 * 0.1175494210692441075487029444849287348827...
    // 10^-37 * 0.1175494350822287507968736537222245677818...
    // 10^-37 * 0.1175494490952133940450443629595204006810...
    check_short_f32!(f32::MIN_POSITIVE => "11754944", -37);

    // 10^-44 * 0
    // 10^-44 * 0.1401298464324817070923729583289916131280...
    // 10^-44 * 0.2802596928649634141847459166579832262560...
    check_short_f32!(ldexp_f32(1.0, -149) => "1", -44);
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
pub fn f32_exact_sanity_test() {
    check_fixed_infz_f32!(0.1             => "100000001490116119384765625",                0);
    check_fixed_infz_f32!(0.5             => "5",                                          0);
    check_fixed_infz_f32!(1.0/3.0         => "3333333432674407958984375",                  0);
    check_fixed_infz_f32!(3.141592        => "31415920257568359375",                       1);
    check_fixed_infz_f32!(3.141592e17     => "314159196796878848",                        18);
    check_fixed_infz_f32!(f32::MAX        => "34028234663852885981170418348451692544",    39);
    check_fixed_f32!(f32::MIN_POSITIVE    => "1175494350822287507968736537222245677819", -37);
    check_fixed_f32!(ldexp_f32(1.0, -149) => "1401298464324817070923729583289916131280", -44);

    // [1], Table 16: Stress Inputs for Converting 24-bit Binary to Decimal, < 1/2 ULP
    check_fixed_one_f32!(12676506.0, -102 => "2",            -23);
    check_fixed_one_f32!(12676506.0, -103 => "12",           -23);
    check_fixed_one_f32!(15445013.0,   86 => "119",           34);
    check_fixed_one_f32!(13734123.0, -138 => "3941",         -34);
    check_fixed_one_f32!(12428269.0, -130 => "91308",        -32);
    check_fixed_one_f32!(15334037.0, -146 => "171900",       -36);
    check_fixed_one_f32!(11518287.0,  -41 => "5237910",       -5);
    check_fixed_one_f32!(12584953.0, -145 => "28216440",     -36);
    check_fixed_one_f32!(15961084.0, -125 => "375243281",    -30);
    check_fixed_one_f32!(14915817.0, -146 => "1672120916",   -36);
    check_fixed_one_f32!(10845484.0, -102 => "21388945814",  -23);
    check_fixed_one_f32!(16431059.0,  -61 => "712583594561", -11);

    // [1], Table 17: Stress Inputs for Converting 24-bit Binary to Decimal, > 1/2 ULP
    check_fixed_one_f32!(16093626.0,   69 => "1",             29);
    check_fixed_one_f32!( 9983778.0,   25 => "34",            15);
    check_fixed_one_f32!(12745034.0,  104 => "259",           39);
    check_fixed_one_f32!(12706553.0,   72 => "6001",          29);
    check_fixed_one_f32!(11005028.0,   45 => "38721",         21);
    check_fixed_one_f32!(15059547.0,   71 => "355584",        29);
    check_fixed_one_f32!(16015691.0,  -99 => "2526831",      -22);
    check_fixed_one_f32!( 8667859.0,   56 => "62458507",      24);
    check_fixed_one_f32!(14855922.0,  -82 => "307213267",    -17);
    check_fixed_one_f32!(14855922.0,  -83 => "1536066333",   -17);
    check_fixed_one_f32!(10144164.0, -110 => "78147796834",  -26);
    check_fixed_one_f32!(13248074.0,   95 => "524810279937",  36);
}

#[test]
pub fn f64_short_sanity_test() {
    // 0.0999999999999999777955395074968691915273...
    // 0.1000000000000000055511151231257827021181...
    // 0.1000000000000000333066907387546962127089...
    check_short_f64!(0.1 => "1", 0);

    // this example is explicitly mentioned in the paper.
    // 10^3 * 0.0999999999999999857891452847979962825775...
    // 10^3 * 0.1 (exact)
    // 10^3 * 0.1000000000000000142108547152020037174224...
    check_short_f64!(100.0 => "1", 3);

    // 0.3333333333333332593184650249895639717578...
    // 0.3333333333333333148296162562473909929394... (1/3 in the default rounding)
    // 0.3333333333333333703407674875052180141210...
    check_short_f64!(1.0/3.0 => "3333333333333333", 0);

    // explicit test case for equally closest representations.
    // Dragon has its own tie-breaking rule; Grisu should fall back.
    // 10^1 * 0.1000007629394531027955395074968691915273...
    // 10^1 * 0.100000762939453125 (exact)
    // 10^1 * 0.1000007629394531472044604925031308084726...
    check_short_f64!(1.00000762939453125 => "10000076293945313", 1);

    // 10^1 * 0.3141591999999999718085064159822650253772...
    // 10^1 * 0.3141592000000000162174274009885266423225...
    // 10^1 * 0.3141592000000000606263483859947882592678...
    check_short_f64!(3.141592 => "3141592", 1);

    // 10^18 * 0.314159199999999936
    // 10^18 * 0.3141592 (exact)
    // 10^18 * 0.314159200000000064
    check_short_f64!(3.141592e17 => "3141592", 18);

    // regression test for decoders
    // 10^20 * 0.18446744073709549568
    // 10^20 * 0.18446744073709551616
    // 10^20 * 0.18446744073709555712
    check_short_f64!(ldexp_f64(1.0, 64) => "18446744073709552", 20);

    // pathological case: high = 10^23 (exact). tie breaking should always prefer that.
    // 10^24 * 0.099999999999999974834176
    // 10^24 * 0.099999999999999991611392
    // 10^24 * 0.100000000000000008388608
    check_short_f64!(1.0e23 => "1", 24);

    // 10^309 * 0.1797693134862315508561243283845062402343...
    // 10^309 * 0.1797693134862315708145274237317043567980...
    // 10^309 * 0.1797693134862315907729305190789024733617...
    check_short_f64!(f64::MAX => "17976931348623157", 309);

    // 10^-307 * 0.2225073858507200889024586876085859887650...
    // 10^-307 * 0.2225073858507201383090232717332404064219...
    // 10^-307 * 0.2225073858507201877155878558578948240788...
    check_short_f64!(f64::MIN_POSITIVE => "22250738585072014", -307);

    // 10^-323 * 0
    // 10^-323 * 0.4940656458412465441765687928682213723650...
    // 10^-323 * 0.9881312916824930883531375857364427447301...
    check_short_f64!(ldexp_f64(1.0, -1074) => "5", -323);
}

#[test]
// This test ends up running what I can only assume is some corner-ish case
// of the `exp2` library function, defined in whatever C runtime we're
// using. In VS 2013 this function apparently had a bug as this test fails
// when linked, but with VS 2015 the bug appears fixed as the test runs just
// fine.
//
// The bug seems to be a difference in return value of `exp2(-1057)`, where
// in VS 2013 it returns a double with the bit pattern 0x2 and in VS 2015 it
// returns 0x20000.
//
// For now just ignore this test entirely on MSVC as it's tested elsewhere
// anyway and we're not super interested in testing each platform's exp2
// implementation.
#[cfg_attr(target_env = "msvc", ignore)]
#[cfg_attr(miri, ignore)] // Miri is too slow
pub fn f64_exact_sanity_test() {
    check_fixed_f64!(0.1               => "1000000000000000055511151231257827021182", 0);
    check_fixed_f64!(0.45              => "4500000000000000111022302462515654042363", 0);
    check_fixed_infz_f64!(0.5          => "5",                                        0);
    check_fixed_f64!(0.95              => "9499999999999999555910790149937383830547", 0);
    check_fixed_infz_f64!(100.0        => "1",                                        3);
    check_fixed_f64!(999.5             => "9995000000000000000000000000000000000000", 3);
    check_fixed_f64!(1.0/3.0           => "3333333333333333148296162562473909929395", 0);
    check_fixed_f64!(3.141592          => "3141592000000000162174274009885266423225", 1);
    check_fixed_infz_f64!(3.141592e17  => "3141592",                                  18);
    check_fixed_infz_f64!(1.0e23       => "99999999999999991611392",                  23);
    check_fixed_f64!(f64::MAX          => "1797693134862315708145274237317043567981", 309);
    check_fixed_f64!(f64::MIN_POSITIVE => "2225073858507201383090232717332404064219", -307);

    check_fixed_f64!(ldexp_f64(1.0, -1074) =>
        "4940656458412465441765687928682213723650\
         5980261432476442558568250067550727020875\
         1865299836361635992379796564695445717730\
         9266567103559397963987747960107818781263\
         0071319031140452784581716784898210368871\
         8636056998730723050006387409153564984387\
         3124733972731696151400317153853980741262\
         3856559117102665855668676818703956031062\
         4931945271591492455329305456544401127480\
         1297099995419319894090804165633245247571\
         4786901472678015935523861155013480352649\
         3472019379026810710749170333222684475333\
         5720832431936092382893458368060106011506\
         1698097530783422773183292479049825247307\
         7637592724787465608477820373446969953364\
         7017972677717585125660551199131504891101\
         4510378627381672509558373897335989936648\
         0994116420570263709027924276754456522908\
         7538682506419718265533447265625", -323);

    // [1], Table 3: Stress Inputs for Converting 53-bit Binary to Decimal, < 1/2 ULP
    check_fixed_one_f64!(8511030020275656.0,  -342 => "9",                       -87);
    check_fixed_one_f64!(5201988407066741.0,  -824 => "46",                     -232);
    check_fixed_one_f64!(6406892948269899.0,   237 => "141",                      88);
    check_fixed_one_f64!(8431154198732492.0,    72 => "3981",                     38);
    check_fixed_one_f64!(6475049196144587.0,    99 => "41040",                    46);
    check_fixed_one_f64!(8274307542972842.0,   726 => "292084",                  235);
    check_fixed_one_f64!(5381065484265332.0,  -456 => "2891946",                -121);
    check_fixed_one_f64!(6761728585499734.0, -1057 => "43787718",               -302);
    check_fixed_one_f64!(7976538478610756.0,   376 => "122770163",               130);
    check_fixed_one_f64!(5982403858958067.0,   377 => "1841552452",              130);
    check_fixed_one_f64!(5536995190630837.0,    93 => "54835744350",              44);
    check_fixed_one_f64!(7225450889282194.0,   710 => "389190181146",            230);
    check_fixed_one_f64!(7225450889282194.0,   709 => "1945950905732",           230);
    check_fixed_one_f64!(8703372741147379.0,   117 => "14460958381605",           52);
    check_fixed_one_f64!(8944262675275217.0, -1001 => "417367747458531",        -285);
    check_fixed_one_f64!(7459803696087692.0,  -707 => "1107950772878888",       -196);
    check_fixed_one_f64!(6080469016670379.0,  -381 => "12345501366327440",       -98);
    check_fixed_one_f64!(8385515147034757.0,   721 => "925031711960365024",      233);
    check_fixed_one_f64!(7514216811389786.0,  -828 => "4198047150284889840",    -233);
    check_fixed_one_f64!(8397297803260511.0,  -345 => "11716315319786511046",    -87);
    check_fixed_one_f64!(6733459239310543.0,   202 => "432810072844612493629",    77);
    check_fixed_one_f64!(8091450587292794.0,  -473 => "3317710118160031081518", -126);

    // [1], Table 4: Stress Inputs for Converting 53-bit Binary to Decimal, > 1/2 ULP
    check_fixed_one_f64!(6567258882077402.0,   952 => "3",                       303);
    check_fixed_one_f64!(6712731423444934.0,   535 => "76",                      177);
    check_fixed_one_f64!(6712731423444934.0,   534 => "378",                     177);
    check_fixed_one_f64!(5298405411573037.0,  -957 => "4350",                   -272);
    check_fixed_one_f64!(5137311167659507.0,  -144 => "23037",                   -27);
    check_fixed_one_f64!(6722280709661868.0,   363 => "126301",                  126);
    check_fixed_one_f64!(5344436398034927.0,  -169 => "7142211",                 -35);
    check_fixed_one_f64!(8369123604277281.0,  -853 => "13934574",               -240);
    check_fixed_one_f64!(8995822108487663.0,  -780 => "141463449",              -218);
    check_fixed_one_f64!(8942832835564782.0,  -383 => "4539277920",              -99);
    check_fixed_one_f64!(8942832835564782.0,  -384 => "22696389598",             -99);
    check_fixed_one_f64!(8942832835564782.0,  -385 => "113481947988",            -99);
    check_fixed_one_f64!(6965949469487146.0,  -249 => "7700366561890",           -59);
    check_fixed_one_f64!(6965949469487146.0,  -250 => "38501832809448",          -59);
    check_fixed_one_f64!(6965949469487146.0,  -251 => "192509164047238",         -59);
    check_fixed_one_f64!(7487252720986826.0,   548 => "6898586531774201",        181);
    check_fixed_one_f64!(5592117679628511.0,   164 => "13076622631878654",        66);
    check_fixed_one_f64!(8887055249355788.0,   665 => "136052020756121240",      217);
    check_fixed_one_f64!(6994187472632449.0,   690 => "3592810217475959676",     224);
    check_fixed_one_f64!(8797576579012143.0,   588 => "89125197712484551899",    193);
    check_fixed_one_f64!(7363326733505337.0,   272 => "558769757362301140950",    98);
    check_fixed_one_f64!(8549497411294502.0,  -448 => "1176257830728540379990", -118);
}

#[test]
pub fn more_short_sanity_test() {
    try_short(
        "inclusive",
        &Decoded64 { mant: 99_999_999_999_999_999, minus: 1, plus: 1, exp: 0, tie_to_even: true },
        "1",
        18,
    );
    try_short(
        "exclusive",
        &Decoded64 { mant: 99_999_999_999_999_999, minus: 1, plus: 1, exp: 0, tie_to_even: false },
        "99999999999999999",
        17,
    );
}

#[test]
pub fn to_short_str_test() {
    assert_eq!(format!("{:}", 0.0), "0");
    assert_eq!(format!("{:+}", 0.0), "+0");
    assert_eq!(format!("{:?}", 0.0), "0.0");
    assert_eq!(format!("{:+?}", 0.0), "+0.0");

    assert_eq!(format!("{:}", -0.0), "-0");
    assert_eq!(format!("{:+}", -0.0), "-0");
    assert_eq!(format!("{:?}", -0.0), "-0.0");
    assert_eq!(format!("{:+?}", -0.0), "-0.0");

    assert_eq!(format!("{:}", 1.0 / 0.0), "inf");
    assert_eq!(format!("{:+}", 1.0 / 0.0), "+inf");
    assert_eq!(format!("{:?}", 1.0 / 0.0), "inf");
    assert_eq!(format!("{:+?}", 1.0 / 0.0), "+inf");

    assert_eq!(format!("{:}", -1.0 / 0.0), "-inf");
    assert_eq!(format!("{:+}", -1.0 / 0.0), "-inf");
    assert_eq!(format!("{:?}", -1.0 / 0.0), "-inf");
    assert_eq!(format!("{:+?}", -1.0 / 0.0), "-inf");

    assert_eq!(format!("{:}", 0.0 / 0.0), "NaN");
    assert_eq!(format!("{:+}", 0.0 / 0.0), "NaN");
    assert_eq!(format!("{:?}", 0.0 / 0.0), "NaN");
    assert_eq!(format!("{:+?}", 0.0 / 0.0), "NaN");

    assert_eq!(format!("{:}", 3.14), "3.14");
    assert_eq!(format!("{:+}", 3.14), "+3.14");
    assert_eq!(format!("{:?}", 3.14), "3.14");
    assert_eq!(format!("{:+?}", 3.14), "+3.14");

    assert_eq!(format!("{:}", -3.14), "-3.14");
    assert_eq!(format!("{:+}", -3.14), "-3.14");
    assert_eq!(format!("{:?}", -3.14), "-3.14");
    assert_eq!(format!("{:+?}", -3.14), "-3.14");

    assert_eq!(format!("{:}", 7.5e-4), "0.00075");
    assert_eq!(format!("{:+}", 7.5e-4), "+0.00075");
    assert_eq!(format!("{:?}", 7.5e-4), "0.00075");
    assert_eq!(format!("{:+?}", 7.5e-4), "+0.00075");

    assert_eq!(format!("{:}", 1.9971e15), "1997100000000000");
    assert_eq!(format!("{:+}", 1.9971e15), "+1997100000000000");
    assert_eq!(format!("{:?}", -1.9971e15), "-1997100000000000.0");
    assert_eq!(format!("{:+?}", -1.9971e15), "-1997100000000000.0");

    // f16
    #[cfg(target_has_reliable_f16)]
    {
        assert_eq!(format!("{:}", f16::MAX), "65500");
        let minf16 = crate::num::ldexp_f16(1.0, -24);
        assert_eq!(format!("{}", minf16), "0.00000006");
    }

    // f32
    assert_eq!(format!("{}", f32::MAX), format!("34028235{:0>31}", ""));
    let minf32 = ldexp_f32(1.0, -149);
    assert_eq!(format!("{}", minf32), format!("0.{:0>44}1", ""));

    // f64
    assert_eq!(format!("{}", f64::MAX), format!("17976931348623157{:0>292}", ""));
    let minf64 = ldexp_f64(1.0, -1074);
    assert_eq!(format!("{}", minf64), format!("0.{:0>323}5", ""));
}

#[test]
pub fn to_short_exp_str_test() {
    assert_eq!(format!("{:E}", 0.0), "0E0");
    assert_eq!(format!("{:+E}", 0.0), "+0E0");
    assert_eq!(format!("{:E}", -0.0), "-0E0");
    assert_eq!(format!("{:+E}", -0.0), "-0E0");

    assert_eq!(format!("{:E}", 1.0 / 0.0), "inf");
    assert_eq!(format!("{:+E}", 1.0 / 0.0), "+inf");
    assert_eq!(format!("{:E}", -1.0 / 0.0), "-inf");
    assert_eq!(format!("{:+E}", -1.0 / 0.0), "-inf");
    assert_eq!(format!("{:E}", 0.0 / 0.0), "NaN");
    assert_eq!(format!("{:+E}", 0.0 / 0.0), "NaN");

    assert_eq!(format!("{:E}", 3.14), "3.14E0");
    assert_eq!(format!("{:+E}", 3.14), "+3.14E0");
    assert_eq!(format!("{:E}", -3.14), "-3.14E0");
    assert_eq!(format!("{:+E}", -3.14), "-3.14E0");

    assert_eq!(format!("{:E}", 0.1), "1E-1");
    assert_eq!(format!("{:+E}", 0.1), "+1E-1");
    assert_eq!(format!("{:E}", -0.1), "-1E-1");
    assert_eq!(format!("{:+E}", -0.1), "-1E-1");

    assert_eq!(format!("{:E}", 7.5e-11), "7.5E-11");
    assert_eq!(format!("{:+E}", 7.5e-11), "+7.5E-11");
    assert_eq!(format!("{:E}", -7.5e-11), "-7.5E-11");
    assert_eq!(format!("{:+E}", -7.5e-11), "-7.5E-11");

    assert_eq!(format!("{:E}", 1.9971e20), "1.9971E20");
    assert_eq!(format!("{:+E}", 1.9971e20), "+1.9971E20");
    assert_eq!(format!("{:E}", -1.9971e20), "-1.9971E20");
    assert_eq!(format!("{:+E}", -1.9971e20), "-1.9971E20");

    // the true value of 1.0e23f64 is less than 10^23, but that shouldn't matter here
    assert_eq!(format!("{:E}", 1.0e23), "1E23");
    assert_eq!(format!("{:+E}", 1.0e23), "+1E23");
    assert_eq!(format!("{:E}", -1.0e23), "-1E23");
    assert_eq!(format!("{:+E}", -1.0e23), "-1E23");

    // f16
    #[cfg(target_has_reliable_f16)]
    {
        assert_eq!(format!("{:e}", f16::MAX), "6.55e4");
        let minf16 = crate::num::ldexp_f16(1.0, -24);
        assert_eq!(format!("{:e}", minf16), "6e-8");
    }

    // f32
    assert_eq!(format!("{:e}", f32::MAX), "3.4028235e38");
    let minf32 = ldexp_f32(1.0, -149);
    assert_eq!(format!("{:e}", minf32), "1e-45");

    // f64
    assert_eq!(format!("{:e}", f64::MAX), "1.7976931348623157e308");
    let minf64 = ldexp_f64(1.0, -1074);
    assert_eq!(format!("{:e}", minf64), "5e-324");
}

#[test]
pub fn to_exact_exp_str_test() {
    assert_eq!(format!("{:.0E}", 0.0), "0E0");
    assert_eq!(format!("{:+.0E}", 0.0), "+0E0");
    assert_eq!(format!("{:.0E}", -0.0), "-0E0");
    assert_eq!(format!("{:+.0E}", -0.0), "-0E0");

    assert_eq!(format!("{:.1E}", 0.0), "0.0E0");
    assert_eq!(format!("{:+.1E}", 0.0), "+0.0E0");
    assert_eq!(format!("{:.1E}", -0.0), "-0.0E0");
    assert_eq!(format!("{:+.1E}", -0.0), "-0.0E0");

    assert_eq!(format!("{:.8E}", 0.0), "0.00000000E0");
    assert_eq!(format!("{:+.8E}", 0.0), "+0.00000000E0");
    assert_eq!(format!("{:.8E}", -0.0), "-0.00000000E0");
    assert_eq!(format!("{:+.8E}", -0.0), "-0.00000000E0");

    assert_eq!(format!("{:.0E}", 1.0 / 0.0), "inf");
    assert_eq!(format!("{:+.1E}", 1.0 / 0.0), "+inf");
    assert_eq!(format!("{:.2E}", -1.0 / 0.0), "-inf");
    assert_eq!(format!("{:+.3E}", -1.0 / 0.0), "-inf");
    assert_eq!(format!("{:.4E}", 0.0 / 0.0), "NaN");
    assert_eq!(format!("{:+.5E}", 0.0 / 0.0), "NaN");

    assert_eq!(format!("{:.0E}", 3.14), "3E0");
    assert_eq!(format!("{:+.0E}", 3.14), "+3E0");
    assert_eq!(format!("{:.0E}", -3.14), "-3E0");
    assert_eq!(format!("{:+.0E}", -3.14), "-3E0");

    assert_eq!(format!("{:.1E}", 3.14), "3.1E0");
    assert_eq!(format!("{:+.1E}", 3.14), "+3.1E0");
    assert_eq!(format!("{:.1E}", -3.14), "-3.1E0");
    assert_eq!(format!("{:+.1E}", -3.14), "-3.1E0");

    assert_eq!(format!("{:.2E}", 3.14), "3.14E0");
    assert_eq!(format!("{:+.2E}", 3.14), "+3.14E0");
    assert_eq!(format!("{:.2E}", -3.14), "-3.14E0");
    assert_eq!(format!("{:+.2E}", -3.14), "-3.14E0");

    assert_eq!(format!("{:.3E}", 3.14), "3.140E0");
    assert_eq!(format!("{:+.3E}", 3.14), "+3.140E0");
    assert_eq!(format!("{:.3E}", -3.14), "-3.140E0");
    assert_eq!(format!("{:+.3E}", -3.14), "-3.140E0");

    assert_eq!(format!("{:.0E}", 0.195), "2E-1");
    assert_eq!(format!("{:.1E}", 0.195), "2.0E-1");
    assert_eq!(format!("{:.2E}", 0.195), "1.95E-1");
    assert_eq!(format!("{:.3E}", 0.195), "1.950E-1");

    assert_eq!(format!("{:.0E}", -0.195), "-2E-1");
    assert_eq!(format!("{:.1E}", -0.195), "-2.0E-1");
    assert_eq!(format!("{:.2E}", -0.195), "-1.95E-1");
    assert_eq!(format!("{:.3E}", -0.195), "-1.950E-1");

    assert_eq!(format!("{:.0E}", 9.5), "1E1");
    assert_eq!(format!("{:.1E}", 9.5), "9.5E0");
    assert_eq!(format!("{:.2E}", 9.5), "9.50E0");

    assert_eq!(format!("{:.0E}", -9.5), "-1E1");
    assert_eq!(format!("{:.1E}", -9.5), "-9.5E0");
    assert_eq!(format!("{:.2E}", -9.5), "-9.50E0");

    assert_eq!(format!("{:.0e}", 1.0e25), "1e25");
    assert_eq!(format!("{:.1e}", 1.0e25), "1.0e25");
    assert_eq!(format!("{:.14e}", 1.0e25), "1.00000000000000e25");
    assert_eq!(format!("{:.15e}", 1.0e25), "1.000000000000000e25");
    assert_eq!(format!("{:.16e}", 1.0e25), "1.0000000000000001e25");
    assert_eq!(format!("{:.17e}", 1.0e25), "1.00000000000000009e25");
    assert_eq!(format!("{:.18e}", 1.0e25), "1.000000000000000091e25");
    assert_eq!(format!("{:.19e}", 1.0e25), "1.0000000000000000906e25");
    assert_eq!(format!("{:.20e}", 1.0e25), "1.00000000000000009060e25");
    assert_eq!(format!("{:.21e}", 1.0e25), "1.000000000000000090597e25");
    assert_eq!(format!("{:.22e}", 1.0e25), "1.0000000000000000905970e25");
    assert_eq!(format!("{:.23e}", 1.0e25), "1.00000000000000009059697e25");
    assert_eq!(format!("{:.24e}", 1.0e25), "1.000000000000000090596966e25");
    assert_eq!(format!("{:.25e}", 1.0e25), "1.0000000000000000905969664e25");
    assert_eq!(format!("{:.26e}", 1.0e25), "1.00000000000000009059696640e25");
    assert_eq!(format!("{:.29e}", 1.0e25), "1.00000000000000009059696640000e25");

    assert_eq!(format!("{:.0e}", 1.0e-6), "1e-6");
    assert_eq!(format!("{:.1e}", 1.0e-6), "1.0e-6");
    assert_eq!(format!("{:.15e}", 1.0e-6), "1.000000000000000e-6");
    assert_eq!(format!("{:.16e}", 1.0e-6), "9.9999999999999995e-7");
    assert_eq!(format!("{:.17e}", 1.0e-6), "9.99999999999999955e-7");
    assert_eq!(format!("{:.18e}", 1.0e-6), "9.999999999999999547e-7");
    assert_eq!(format!("{:.19e}", 1.0e-6), "9.9999999999999995475e-7");
    assert_eq!(format!("{:.29e}", 1.0e-6), "9.99999999999999954748111825886e-7");
    assert_eq!(format!("{:.39e}", 1.0e-6), "9.999999999999999547481118258862586856139e-7");
    assert_eq!(
        format!("{:.49e}", 1.0e-6),
        "9.9999999999999995474811182588625868561393872369081e-7"
    );
    assert_eq!(
        format!("{:.59e}", 1.0e-6),
        "9.99999999999999954748111825886258685613938723690807819366455e-7"
    );
    assert_eq!(
        format!("{:.69e}", 1.0e-6),
        "9.999999999999999547481118258862586856139387236908078193664550781250000e-7"
    );

    // f16
    #[cfg(target_has_reliable_f16)]
    {
        assert_eq!(format!("{:.0e}", f16::MAX), "7e4");
        assert_eq!(format!("{:.1e}", f16::MAX), "6.6e4");
        assert_eq!(format!("{:.3e}", f16::MAX), "6.550e4");
        assert_eq!(format!("{:.4e}", f16::MAX), "6.5504e4");
        assert_eq!(format!("{:.5e}", f16::MAX), "6.55040e4");
        assert_eq!(format!("{:.15e}", f16::MAX), "6.550400000000000e4");

        let minf16 = crate::num::ldexp_f16(1.0, -24);
        assert_eq!(format!("{:.0e}", minf16), "6e-8");
        assert_eq!(format!("{:.1e}", minf16), "6.0e-8");
        assert_eq!(format!("{:.3e}", minf16), "5.960e-8");
        assert_eq!(format!("{:.7e}", minf16), "5.9604645e-8");
        assert_eq!(format!("{:.15e}", minf16), "5.960464477539062e-8");
        assert_eq!(format!("{:.16e}", minf16), "5.9604644775390625e-8");
        assert_eq!(format!("{:.17e}", minf16), "5.96046447753906250e-8");
        assert_eq!(format!("{:.23e}", minf16), "5.96046447753906250000000e-8");
    }

    // f32
    assert_eq!(format!("{:.0e}", f32::MAX), "3e38");
    assert_eq!(format!("{:.1e}", f32::MAX), "3.4e38");
    assert_eq!(format!("{:.3e}", f32::MAX), "3.403e38");
    assert_eq!(format!("{:.7e}", f32::MAX), "3.4028235e38");
    assert_eq!(format!("{:.15e}", f32::MAX), "3.402823466385289e38");
    assert_eq!(format!("{:.31e}", f32::MAX), "3.4028234663852885981170418348452e38");
    assert_eq!(
        format!("{:.63e}", f32::MAX),
        "3.402823466385288598117041834845169254400000000000000000000000000e38"
    );

    let minf32 = ldexp_f32(1.0, -149);
    assert_eq!(format!("{:.0e}", minf32), "1e-45");
    assert_eq!(format!("{:.1e}", minf32), "1.4e-45");
    assert_eq!(format!("{:.3e}", minf32), "1.401e-45");
    assert_eq!(format!("{:.7e}", minf32), "1.4012985e-45");
    assert_eq!(format!("{:.15e}", minf32), "1.401298464324817e-45");
    assert_eq!(format!("{:.31e}", minf32), "1.4012984643248170709237295832899e-45");
    assert_eq!(
        format!("{:.63e}", minf32),
        "1.401298464324817070923729583289916131280261941876515771757068284e-45"
    );
    assert_eq!(
        format!("{:.127e}", minf32),
        "1.401298464324817070923729583289916131280261941876515771757068283\
         8897910826858606014866381883621215820312500000000000000000000000e-45"
    );

    if cfg!(miri) {
        // Miri is too slow
        return;
    }

    // f64
    assert_eq!(format!("{:.0e}", f64::MAX), "2e308");
    assert_eq!(format!("{:.1e}", f64::MAX), "1.8e308");
    assert_eq!(format!("{:.3e}", f64::MAX), "1.798e308");
    assert_eq!(format!("{:.7e}", f64::MAX), "1.7976931e308");
    assert_eq!(format!("{:.15e}", f64::MAX), "1.797693134862316e308");
    assert_eq!(format!("{:.31e}", f64::MAX), "1.7976931348623157081452742373170e308");
    assert_eq!(
        format!("{:.63e}", f64::MAX),
        "1.797693134862315708145274237317043567980705675258449965989174768e308"
    );
    assert_eq!(
        format!("{:.127e}", f64::MAX),
        "1.797693134862315708145274237317043567980705675258449965989174768\
         0315726078002853876058955863276687817154045895351438246423432133e308"
    );
    assert_eq!(
        format!("{:.255e}", f64::MAX),
        "1.797693134862315708145274237317043567980705675258449965989174768\
         0315726078002853876058955863276687817154045895351438246423432132\
         6889464182768467546703537516986049910576551282076245490090389328\
         9440758685084551339423045832369032229481658085593321233482747978e308"
    );
    assert_eq!(
        format!("{:.511e}", f64::MAX),
        "1.797693134862315708145274237317043567980705675258449965989174768\
         0315726078002853876058955863276687817154045895351438246423432132\
         6889464182768467546703537516986049910576551282076245490090389328\
         9440758685084551339423045832369032229481658085593321233482747978\
         2620414472316873817718091929988125040402618412485836800000000000\
         0000000000000000000000000000000000000000000000000000000000000000\
         0000000000000000000000000000000000000000000000000000000000000000\
         0000000000000000000000000000000000000000000000000000000000000000e308"
    );

    let minf64 = ldexp_f64(1.0, -1074);
    assert_eq!(format!("{:.0e}", minf64), "5e-324");
    assert_eq!(format!("{:.1e}", minf64), "4.9e-324");
    assert_eq!(format!("{:.3e}", minf64), "4.941e-324");
    assert_eq!(format!("{:.7e}", minf64), "4.9406565e-324");
    assert_eq!(format!("{:.15e}", minf64), "4.940656458412465e-324");
    assert_eq!(format!("{:.31e}", minf64), "4.9406564584124654417656879286822e-324");
    assert_eq!(
        format!("{:.63e}", minf64),
        "4.940656458412465441765687928682213723650598026143247644255856825e-324"
    );
    assert_eq!(
        format!("{:.127e}", minf64),
        "4.940656458412465441765687928682213723650598026143247644255856825\
         0067550727020875186529983636163599237979656469544571773092665671e-324"
    );
    assert_eq!(
        format!("{:.255e}", minf64),
        "4.940656458412465441765687928682213723650598026143247644255856825\
         0067550727020875186529983636163599237979656469544571773092665671\
         0355939796398774796010781878126300713190311404527845817167848982\
         1036887186360569987307230500063874091535649843873124733972731696e-324"
    );
    assert_eq!(
        format!("{:.511e}", minf64),
        "4.940656458412465441765687928682213723650598026143247644255856825\
         0067550727020875186529983636163599237979656469544571773092665671\
         0355939796398774796010781878126300713190311404527845817167848982\
         1036887186360569987307230500063874091535649843873124733972731696\
         1514003171538539807412623856559117102665855668676818703956031062\
         4931945271591492455329305456544401127480129709999541931989409080\
         4165633245247571478690147267801593552386115501348035264934720193\
         7902681071074917033322268447533357208324319360923828934583680601e-324"
    );
    assert_eq!(
        format!("{:.1023e}", minf64),
        "4.940656458412465441765687928682213723650598026143247644255856825\
         0067550727020875186529983636163599237979656469544571773092665671\
         0355939796398774796010781878126300713190311404527845817167848982\
         1036887186360569987307230500063874091535649843873124733972731696\
         1514003171538539807412623856559117102665855668676818703956031062\
         4931945271591492455329305456544401127480129709999541931989409080\
         4165633245247571478690147267801593552386115501348035264934720193\
         7902681071074917033322268447533357208324319360923828934583680601\
         0601150616980975307834227731832924790498252473077637592724787465\
         6084778203734469699533647017972677717585125660551199131504891101\
         4510378627381672509558373897335989936648099411642057026370902792\
         4276754456522908753868250641971826553344726562500000000000000000\
         0000000000000000000000000000000000000000000000000000000000000000\
         0000000000000000000000000000000000000000000000000000000000000000\
         0000000000000000000000000000000000000000000000000000000000000000\
         0000000000000000000000000000000000000000000000000000000000000000e-324"
    );

    // very large output
    assert_eq!(format!("{:.49999e}", 0.0), format!("0.{:0>49999}e0", ""));
    assert_eq!(format!("{:.49999e}", 10.0), format!("1.{:0>49999}e1", ""));
    assert_eq!(format!("{:.49999e}", 1.0), format!("1.{:0>49999}e0", ""));
    assert_eq!(
        format!("{:.49999e}", 0.1),
        format!("1.000000000000000055511151231257827021181583404541015625{:0>49945}e-1", "",)
    );
    assert_eq!(
        format!("{:.49999e}", 1.0e-20),
        format!(
            "9.999999999999999451532714542095716517295037027873924471077157760\
             66783064379706047475337982177734375{:0>49901}e-21",
            "",
        )
    );
}

#[test]
pub fn to_exact_fixed_str_test() {
    assert_eq!(format!("{:.0}", 3.14), "3");
    assert_eq!(format!("{:.1}", 3.14), "3.1");
    assert_eq!(format!("{:.2}", 3.14), "3.14");
    assert_eq!(format!("{:.3}", 3.14), "3.140");
    assert_eq!(format!("{:.4}", 3.14), "3.1400");

    assert_eq!(format!("{:.0}", 0.195), "0");
    assert_eq!(format!("{:.1}", 0.195), "0.2");
    assert_eq!(format!("{:.2}", 0.195), "0.20");
    assert_eq!(format!("{:+.2}", 0.195), "+0.20");
    assert_eq!(format!("{:+.2}", -0.195), "-0.20");
    assert_eq!(format!("{:.2}", -0.195), "-0.20");
    assert_eq!(format!("{:.3}", 0.195), "0.195");
    assert_eq!(format!("{:+.3}", 0.195), "+0.195");
    assert_eq!(format!("{:+.3}", -0.195), "-0.195");
    assert_eq!(format!("{:.3}", -0.195), "-0.195");
    assert_eq!(format!("{:.4}", 0.195), "0.1950");
    assert_eq!(format!("{:+.4}", 0.195), "+0.1950");
    assert_eq!(format!("{:+.4}", -0.195), "-0.1950");
    assert_eq!(format!("{:.4}", -0.195), "-0.1950");

    assert_eq!(format!("{:.0}", 999.5), "1000");
    assert_eq!(format!("{:.1}", 999.5), "999.5");
    assert_eq!(format!("{:.2}", 999.5), "999.50");
    assert_eq!(format!("{:.3}", 999.5), "999.500");
    assert_eq!(format!("{:.30}", 999.5), "999.500000000000000000000000000000");

    assert_eq!(format!("{:.0}", 0.5), "0");
    assert_eq!(format!("{:.1}", 0.5), "0.5");
    assert_eq!(format!("{:.2}", 0.5), "0.50");
    assert_eq!(format!("{:.3}", 0.5), "0.500");

    assert_eq!(format!("{:.0}", 0.95), "1");
    assert_eq!(format!("{:.1}", 0.95), "0.9"); // because it really is less than 0.95
    assert_eq!(format!("{:.2}", 0.95), "0.95");
    assert_eq!(format!("{:.3}", 0.95), "0.950");
    assert_eq!(format!("{:.10}", 0.95), "0.9500000000");
    assert_eq!(format!("{:.30}", 0.95), "0.949999999999999955591079014994");

    assert_eq!(format!("{:.0}", 0.095), "0");
    assert_eq!(format!("{:.1}", 0.095), "0.1");
    assert_eq!(format!("{:.2}", 0.095), "0.10");
    assert_eq!(format!("{:.3}", 0.095), "0.095");
    assert_eq!(format!("{:.4}", 0.095), "0.0950");
    assert_eq!(format!("{:.10}", 0.095), "0.0950000000");
    assert_eq!(format!("{:.30}", 0.095), "0.095000000000000001110223024625");

    assert_eq!(format!("{:.0}", 0.0095), "0");
    assert_eq!(format!("{:.1}", 0.0095), "0.0");
    assert_eq!(format!("{:.2}", 0.0095), "0.01");
    assert_eq!(format!("{:.3}", 0.0095), "0.009"); // because it really is less than 0.0095
    assert_eq!(format!("{:.4}", 0.0095), "0.0095");
    assert_eq!(format!("{:.5}", 0.0095), "0.00950");
    assert_eq!(format!("{:.10}", 0.0095), "0.0095000000");
    assert_eq!(format!("{:.30}", 0.0095), "0.009499999999999999764077607267");

    assert_eq!(format!("{:.0}", 7.5e-11), "0");
    assert_eq!(format!("{:.3}", 7.5e-11), "0.000");
    assert_eq!(format!("{:.10}", 7.5e-11), "0.0000000001");
    assert_eq!(format!("{:.11}", 7.5e-11), "0.00000000007"); // ditto
    assert_eq!(format!("{:.12}", 7.5e-11), "0.000000000075");
    assert_eq!(format!("{:.13}", 7.5e-11), "0.0000000000750");
    assert_eq!(format!("{:.20}", 7.5e-11), "0.00000000007500000000");
    assert_eq!(format!("{:.30}", 7.5e-11), "0.000000000074999999999999999501");

    assert_eq!(format!("{:.0}", 1.0e25), "10000000000000000905969664");
    assert_eq!(format!("{:.1}", 1.0e25), "10000000000000000905969664.0");
    assert_eq!(format!("{:.3}", 1.0e25), "10000000000000000905969664.000");

    assert_eq!(format!("{:.0}", 1.0e-6), "0");
    assert_eq!(format!("{:.3}", 1.0e-6), "0.000");
    assert_eq!(format!("{:.6}", 1.0e-6), "0.000001");
    assert_eq!(format!("{:.9}", 1.0e-6), "0.000001000");
    assert_eq!(format!("{:.12}", 1.0e-6), "0.000001000000");
    assert_eq!(format!("{:.22}", 1.0e-6), "0.0000010000000000000000");
    assert_eq!(format!("{:.23}", 1.0e-6), "0.00000099999999999999995");
    assert_eq!(format!("{:.24}", 1.0e-6), "0.000000999999999999999955");
    assert_eq!(format!("{:.25}", 1.0e-6), "0.0000009999999999999999547");
    assert_eq!(format!("{:.35}", 1.0e-6), "0.00000099999999999999995474811182589");
    assert_eq!(format!("{:.45}", 1.0e-6), "0.000000999999999999999954748111825886258685614");
    assert_eq!(
        format!("{:.55}", 1.0e-6),
        "0.0000009999999999999999547481118258862586856139387236908"
    );
    assert_eq!(
        format!("{:.65}", 1.0e-6),
        "0.00000099999999999999995474811182588625868561393872369080781936646"
    );
    assert_eq!(
        format!("{:.75}", 1.0e-6),
        "0.000000999999999999999954748111825886258685613938723690807819366455078125000"
    );

    // f16
    #[cfg(target_has_reliable_f16)]
    {
        assert_eq!(format!("{:.0}", f16::MAX), "65504");
        assert_eq!(format!("{:.1}", f16::MAX), "65504.0");
        assert_eq!(format!("{:.2}", f16::MAX), "65504.00");
    }
    // f32
    assert_eq!(format!("{:.0}", f32::MAX), "340282346638528859811704183484516925440");
    assert_eq!(format!("{:.1}", f32::MAX), "340282346638528859811704183484516925440.0");
    assert_eq!(format!("{:.2}", f32::MAX), "340282346638528859811704183484516925440.00");

    if cfg!(miri) {
        // Miri is too slow
        return;
    }

    #[cfg(target_has_reliable_f16)]
    {
        let minf16 = crate::num::ldexp_f16(1.0, -24);
        assert_eq!(format!("{:.0}", minf16), "0");
        assert_eq!(format!("{:.1}", minf16), "0.0");
        assert_eq!(format!("{:.2}", minf16), "0.00");
        assert_eq!(format!("{:.4}", minf16), "0.0000");
        assert_eq!(format!("{:.8}", minf16), "0.00000006");
        assert_eq!(format!("{:.10}", minf16), "0.0000000596");
        assert_eq!(format!("{:.15}", minf16), "0.000000059604645");
        assert_eq!(format!("{:.20}", minf16), "0.00000005960464477539");
        assert_eq!(format!("{:.24}", minf16), "0.000000059604644775390625");
        assert_eq!(format!("{:.32}", minf16), "0.00000005960464477539062500000000");
    }

    let minf32 = ldexp_f32(1.0, -149);
    assert_eq!(format!("{:.0}", minf32), "0");
    assert_eq!(format!("{:.1}", minf32), "0.0");
    assert_eq!(format!("{:.2}", minf32), "0.00");
    assert_eq!(format!("{:.4}", minf32), "0.0000");
    assert_eq!(format!("{:.8}", minf32), "0.00000000");
    assert_eq!(format!("{:.16}", minf32), "0.0000000000000000");
    assert_eq!(format!("{:.32}", minf32), "0.00000000000000000000000000000000");
    assert_eq!(
        format!("{:.64}", minf32),
        "0.0000000000000000000000000000000000000000000014012984643248170709"
    );
    assert_eq!(
        format!("{:.128}", minf32),
        "0.0000000000000000000000000000000000000000000014012984643248170709\
         2372958328991613128026194187651577175706828388979108268586060149"
    );
    assert_eq!(
        format!("{:.256}", minf32),
        "0.0000000000000000000000000000000000000000000014012984643248170709\
         2372958328991613128026194187651577175706828388979108268586060148\
         6638188362121582031250000000000000000000000000000000000000000000\
         0000000000000000000000000000000000000000000000000000000000000000"
    );

    assert_eq!(
        format!("{:.0}", f64::MAX),
        "1797693134862315708145274237317043567980705675258449965989174768\
         0315726078002853876058955863276687817154045895351438246423432132\
         6889464182768467546703537516986049910576551282076245490090389328\
         9440758685084551339423045832369032229481658085593321233482747978\
         26204144723168738177180919299881250404026184124858368"
    );
    assert_eq!(
        format!("{:.10}", f64::MAX),
        "1797693134862315708145274237317043567980705675258449965989174768\
         0315726078002853876058955863276687817154045895351438246423432132\
         6889464182768467546703537516986049910576551282076245490090389328\
         9440758685084551339423045832369032229481658085593321233482747978\
         26204144723168738177180919299881250404026184124858368.0000000000"
    );

    let minf64 = ldexp_f64(1.0, -1074);
    assert_eq!(format!("{:.0}", minf64), "0");
    assert_eq!(format!("{:.1}", minf64), "0.0");
    assert_eq!(format!("{:.10}", minf64), "0.0000000000");
    assert_eq!(
        format!("{:.100}", minf64),
        "0.0000000000000000000000000000000000000000000000000000000000000000\
         000000000000000000000000000000000000"
    );
    assert_eq!(
        format!("{:.1000}", minf64),
        "0.0000000000000000000000000000000000000000000000000000000000000000\
         0000000000000000000000000000000000000000000000000000000000000000\
         0000000000000000000000000000000000000000000000000000000000000000\
         0000000000000000000000000000000000000000000000000000000000000000\
         0000000000000000000000000000000000000000000000000000000000000000\
         0004940656458412465441765687928682213723650598026143247644255856\
         8250067550727020875186529983636163599237979656469544571773092665\
         6710355939796398774796010781878126300713190311404527845817167848\
         9821036887186360569987307230500063874091535649843873124733972731\
         6961514003171538539807412623856559117102665855668676818703956031\
         0624931945271591492455329305456544401127480129709999541931989409\
         0804165633245247571478690147267801593552386115501348035264934720\
         1937902681071074917033322268447533357208324319360923828934583680\
         6010601150616980975307834227731832924790498252473077637592724787\
         4656084778203734469699533647017972677717585125660551199131504891\
         1014510378627381672509558373897335989937"
    );

    // very large output
    assert_eq!(format!("{:.50000}", 0.0), format!("0.{:0>50000}", ""));
    assert_eq!(format!("{:.50000}", 10.0), format!("10.{:0>50000}", ""));
    assert_eq!(format!("{:.50000}", 1.0), format!("1.{:0>50000}", ""));
    assert_eq!(
        format!("{:.50000}", 0.1),
        format!("0.1000000000000000055511151231257827021181583404541015625{:0>49945}", "",)
    );
    assert_eq!(
        format!("{:.50000}", 1.0e-20),
        format!(
            "0.0000000000000000000099999999999999994515327145420957165172950370\
             2787392447107715776066783064379706047475337982177734375{:0>49881}",
            "",
        )
    );
}
