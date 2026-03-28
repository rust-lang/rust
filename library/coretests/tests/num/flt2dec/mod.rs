use core::num::imp::flt2dec;
use core::num::imp::flt2dec::{
    DecodableFloat, Decoded, FullDecoded, MAX_SIG_DIGITS, Sign, decode, round_up, to_exact_exp_str,
    to_exact_fixed_str, to_shortest_exp_str, to_shortest_str,
};
use core::num::imp::fmt::{Formatted, Part};
use std::mem::MaybeUninit;

use Sign::{Minus, MinusPlus};

use crate::num::{ldexp_f32, ldexp_f64};

mod estimator;
mod strategy {
    mod dragon;
    mod grisu;
}
mod random;

fn decode_finite<T: DecodableFloat>(v: T) -> Decoded {
    match decode(v).1 {
        FullDecoded::Finite(decoded) => decoded,
        full_decoded => panic!("expected finite, got {full_decoded:?} instead for {v:?}"),
    }
}

/// Assert format_short outcome.
macro_rules! check_short {
    ($v:expr => $digits:expr, $pow10:expr) => {
        let label = stringify!($v);
        let dec = decode_finite($v);
        let want: (&[u8], i16) = ($digits, $pow10);

        let mut buf = [MaybeUninit::new(b'_'); MAX_SIG_DIGITS];
        let got = flt2dec::format_short(&dec, &mut buf);
        assert_eq!(got, want, "short format of {label}");
    };
}

/// Assert format_fixed outcome with a specific resolution.
macro_rules! check_resolution {
    ($v:expr, $resolution:expr => $digits:expr, $pow10:expr) => {
        let label = stringify!($v);
        let dec = decode_finite($v);
        let resolution: i16 = $resolution;
        let want: (&[u8], i16) = ($digits, $pow10);

        let mut large_buf = [MaybeUninit::new(b'_'); 1024];
        let got = flt2dec::format_fixed(&dec, &mut large_buf, resolution);
        assert_eq!(got, want, "short format of {label}");
    };
}

/// Assert format_fixed outcome with a matching buffer size.
macro_rules! check_match {
    ($v:expr => $digits:expr, $pow10:expr) => {
        let label = stringify!($v);
        let dec = decode_finite($v);
        let want: (&[u8], i16) = ($digits, $pow10);

        let mut buf = [MaybeUninit::new(b'_'); 1024];
        let mut matched_size = &mut buf[..want.0.len()];
        let unlimited_resolution = i16::MIN;
        let got = flt2dec::format_fixed(&dec, &mut matched_size, unlimited_resolution);
        assert_eq!(got, want, "fixed format of {label}");
    };
}

/// Assert format_fixed outcome including variants at a lower resolution.
/// An equals suffix ("=") on `$digits` triggers testing of infinite zeroes.
macro_rules! check_fixed_mix {
    ($v:expr => $digits:expr, $pow10:expr) => (
        let (digits, inf_zero): (&[u8], bool) = {
            let d: &[u8] = $digits;
            match d.strip_suffix(b"=".as_slice()) {
                Some(s) => (s, true),
                None => (d, false),
            }
        };
        let pow10: i16 = $pow10;

        // Verify base assertion.
        check_match!($v => &digits, pow10);
        check_resolution!($v, pow10 - digits.len() as i16 => &digits, pow10);

        let mut want_buf = [b'_'; 1024];

        // Verify at a lower resolution with substrings of `$digits`.
        for i in 1..(digits.len() - 1) {
            want_buf[..i].copy_from_slice(&digits[..i]);
            let mut want_pow10 = pow10;
            if digits[i] >= b'5' {
                // Include rounding-to-even case (with equals suffix).
                if !(digits[i - 1] & 1 == 0
                    && digits[i] == b'5'
                    && digits[i + 1] == b'=')
                {
                    if let Some(_) = round_up(&mut want_buf[..i]) {
                        // All '9's got rounded up to an extra digit.
                        want_pow10 += 1;
                    }
                }
            }

            check_match!($v => &want_buf[..i], want_pow10);
            check_resolution!($v, want_pow10 - i as i16 => &want_buf[..i], want_pow10);
        }

        // Verify digit loss by resolution.
        let skip_first = digits[0] > b'5';
        if skip_first {
            check_resolution!($v, pow10 => b"1", pow10 + 1);
        }
        for i in (skip_first as i16)..-10 {
            // Digit loss must still maintain the exponent.
            check_resolution!($v, pow10 - i => b"", pow10);
        }

        // Verify infinite zero digits.
        if inf_zero {
            let zero_trail = b"0000000";
            let trail_range = digits.len()..(digits.len() + zero_trail.len());
            want_buf[..trail_range.start].copy_from_slice(digits);
            want_buf[trail_range.clone()].copy_from_slice(&zero_trail[..]);
            for last in trail_range {
                let end = last + 1;
                check_match!($v => &want_buf[..end], pow10);
                check_resolution!($v, pow10 - end as i16 => &want_buf[..end], pow10);
            }
        }
    )
}

macro_rules! check_coef_pow2 {
    ($coef:expr, $pow2:expr => $digits:expr, $pow10:expr) => (
         check_match!($coef * $pow2.exp2() => $digits, $pow10);
         check_resolution!($coef * $pow2.exp2(), $pow10 - $digits.len() as i16 => $digits, $pow10);
    )
}

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
fn f16_short_sanity_test() {
    // 0.0999145507813
    // 0.0999755859375
    // 0.100036621094
    check_short!(0.1f16 => b"1", 0);

    // 0.3330078125
    // 0.333251953125 (1/3 in the default rounding)
    // 0.33349609375
    check_short!(1.0f16/3.0 => b"3333", 0);

    // 10^1 * 0.3138671875
    // 10^1 * 0.3140625
    // 10^1 * 0.3142578125
    check_short!(3.14f16 => b"314", 1);

    // 10^18 * 0.31415916243714048
    // 10^18 * 0.314159196796878848
    // 10^18 * 0.314159231156617216
    check_short!(3.1415e4f16 => b"3141", 5);

    // regression test for decoders
    // 10^2 * 0.31984375
    // 10^2 * 0.32
    // 10^2 * 0.3203125
    check_short!(crate::num::ldexp_f16(1.0, 5) => b"32", 2);

    // 10^5 * 0.65472
    // 10^5 * 0.65504
    // 10^5 * 0.65536
    check_short!(f16::MAX => b"655", 5);

    // 10^-4 * 0.60975551605224609375
    // 10^-4 * 0.6103515625
    // 10^-4 * 0.61094760894775390625
    check_short!(f16::MIN_POSITIVE => b"6104", -4);

    // 10^-9 * 0
    // 10^-9 * 0.59604644775390625
    // 10^-8 * 0.11920928955078125
    check_short!(crate::num::ldexp_f16(1.0, -24) => b"6", -7);
}

#[test]
#[cfg(target_has_reliable_f16)]
fn f16_fixed_sanity_test() {
    let minf16 = crate::num::ldexp_f16(1.0, -24);

    check_fixed_mix!(0.1f16            => b"999755859375=", -1);
    check_fixed_mix!(0.5f16            => b"5=", 0);
    check_fixed_mix!(1.0f16/3.0        => b"333251953125=", 0);
    check_fixed_mix!(3.141f16          => b"3140625=", 1);
    check_fixed_mix!(3.141e4f16        => b"31408=", 5);
    check_fixed_mix!(f16::MAX          => b"65504=", 5);
    check_fixed_mix!(f16::MIN_POSITIVE => b"6103515625=", -4);
    check_fixed_mix!(minf16            => b"59604644775390625", -7);

    // FIXME(f16): these should gain the check_coef_pow2 tests like `f32` and `f64` have,
    // but these values are not easy to generate. The algorithm from the Paxon paper [1] needs
    // to be adapted to binary16.
}

#[test]
fn f32_short_sanity_test() {
    // 0.0999999940395355224609375
    // 0.100000001490116119384765625
    // 0.10000000894069671630859375
    check_short!(0.1f32 => b"1", 0);

    // 0.333333313465118408203125
    // 0.3333333432674407958984375 (1/3 in the default rounding)
    // 0.33333337306976318359375
    check_short!(1.0f32/3.0 => b"33333334", 0);

    // 10^1 * 0.31415917873382568359375
    // 10^1 * 0.31415920257568359375
    // 10^1 * 0.31415922641754150390625
    check_short!(3.141592f32 => b"3141592", 1);

    // 10^18 * 0.31415916243714048
    // 10^18 * 0.314159196796878848
    // 10^18 * 0.314159231156617216
    check_short!(3.141592e17f32 => b"3141592", 18);

    // regression test for decoders
    // 10^8 * 0.3355443
    // 10^8 * 0.33554432
    // 10^8 * 0.33554436
    check_short!(ldexp_f32(1.0, 25) => b"33554432", 8);

    // 10^39 * 0.340282326356119256160033759537265639424
    // 10^39 * 0.34028234663852885981170418348451692544
    // 10^39 * 0.340282366920938463463374607431768211456
    check_short!(f32::MAX => b"34028235", 39);

    // 10^-37 * 0.1175494210692441075487029444849287348827...
    // 10^-37 * 0.1175494350822287507968736537222245677818...
    // 10^-37 * 0.1175494490952133940450443629595204006810...
    check_short!(f32::MIN_POSITIVE => b"11754944", -37);

    // 10^-44 * 0
    // 10^-44 * 0.1401298464324817070923729583289916131280...
    // 10^-44 * 0.2802596928649634141847459166579832262560...
    check_short!(ldexp_f32(1.0, -149) => b"1", -44);
}

#[test]
fn f32_fixed_sanity_test() {
    let minf32 = ldexp_f32(1.0, -149);

    check_fixed_mix!(0.1f32            => b"100000001490116119384765625=", 0);
    check_fixed_mix!(0.5f32            => b"5=", 0);
    check_fixed_mix!(1.0f32/3.0        => b"3333333432674407958984375=", 0);
    check_fixed_mix!(3.141592f32       => b"31415920257568359375=", 1);
    check_fixed_mix!(3.141592e17f32    => b"314159196796878848=", 18);
    check_fixed_mix!(f32::MAX          => b"34028234663852885981170418348451692544=", 39);
    check_fixed_mix!(f32::MIN_POSITIVE => b"1175494350822287507968736537222245677819", -37);
    check_fixed_mix!(minf32            => b"1401298464324817070923729583289916131280", -44);

    // [1], Table 16: Stress Inputs for Converting 24-bit Binary to Decimal, < 1/2 ULP
    check_coef_pow2!(12676506_f32, -102_f32 => b"2",            -23);
    check_coef_pow2!(12676506_f32, -103_f32 => b"12",           -23);
    check_coef_pow2!(15445013_f32,   86_f32 => b"119",           34);
    check_coef_pow2!(13734123_f32, -138_f32 => b"3941",         -34);
    check_coef_pow2!(12428269_f32, -130_f32 => b"91308",        -32);
    check_coef_pow2!(15334037_f32, -146_f32 => b"171900",       -36);
    check_coef_pow2!(11518287_f32,  -41_f32 => b"5237910",       -5);
    check_coef_pow2!(12584953_f32, -145_f32 => b"28216440",     -36);
    check_coef_pow2!(15961084_f32, -125_f32 => b"375243281",    -30);
    check_coef_pow2!(14915817_f32, -146_f32 => b"1672120916",   -36);
    check_coef_pow2!(10845484_f32, -102_f32 => b"21388945814",  -23);
    check_coef_pow2!(16431059_f32,  -61_f32 => b"712583594561", -11);

    // [1], Table 17: Stress Inputs for Converting 24-bit Binary to Decimal, > 1/2 ULP
    check_coef_pow2!(16093626_f32,   69_f32 => b"1",             29);
    check_coef_pow2!( 9983778_f32,   25_f32 => b"34",            15);
    check_coef_pow2!(12745034_f32,  104_f32 => b"259",           39);
    check_coef_pow2!(12706553_f32,   72_f32 => b"6001",          29);
    check_coef_pow2!(11005028_f32,   45_f32 => b"38721",         21);
    check_coef_pow2!(15059547_f32,   71_f32 => b"355584",        29);
    check_coef_pow2!(16015691_f32,  -99_f32 => b"2526831",      -22);
    check_coef_pow2!( 8667859_f32,   56_f32 => b"62458507",      24);
    check_coef_pow2!(14855922_f32,  -82_f32 => b"307213267",    -17);
    check_coef_pow2!(14855922_f32,  -83_f32 => b"1536066333",   -17);
    check_coef_pow2!(10144164_f32, -110_f32 => b"78147796834",  -26);
    check_coef_pow2!(13248074_f32,   95_f32 => b"524810279937",  36);
}

#[test]
fn f64_short_sanity_test() {
    // 0.0999999999999999777955395074968691915273...
    // 0.1000000000000000055511151231257827021181...
    // 0.1000000000000000333066907387546962127089...
    check_short!(0.1f64 => b"1", 0);

    // this example is explicitly mentioned in the paper.
    // 10^3 * 0.0999999999999999857891452847979962825775...
    // 10^3 * 0.1 (exact)
    // 10^3 * 0.1000000000000000142108547152020037174224...
    check_short!(100.0f64 => b"1", 3);

    // 0.3333333333333332593184650249895639717578...
    // 0.3333333333333333148296162562473909929394... (1/3 in the default rounding)
    // 0.3333333333333333703407674875052180141210...
    check_short!(1.0f64/3.0 => b"3333333333333333", 0);

    // explicit test case for equally closest representations.
    // Dragon has its own tie-breaking rule; Grisu should fall back.
    // 10^1 * 0.1000007629394531027955395074968691915273...
    // 10^1 * 0.100000762939453125 (exact)
    // 10^1 * 0.1000007629394531472044604925031308084726...
    check_short!(1.00000762939453125f64 => b"10000076293945313", 1);

    // 10^1 * 0.3141591999999999718085064159822650253772...
    // 10^1 * 0.3141592000000000162174274009885266423225...
    // 10^1 * 0.3141592000000000606263483859947882592678...
    check_short!(3.141592f64 => b"3141592", 1);

    // 10^18 * 0.314159199999999936
    // 10^18 * 0.3141592 (exact)
    // 10^18 * 0.314159200000000064
    check_short!(3.141592e17f64 => b"3141592", 18);

    // regression test for decoders
    // 10^20 * 0.18446744073709549568
    // 10^20 * 0.18446744073709551616
    // 10^20 * 0.18446744073709555712
    check_short!(ldexp_f64(1.0, 64) => b"18446744073709552", 20);

    // pathological case: high = 10^23 (exact). tie breaking should always prefer that.
    // 10^24 * 0.099999999999999974834176
    // 10^24 * 0.099999999999999991611392
    // 10^24 * 0.100000000000000008388608
    check_short!(1.0e23f64 => b"1", 24);

    // 10^309 * 0.1797693134862315508561243283845062402343...
    // 10^309 * 0.1797693134862315708145274237317043567980...
    // 10^309 * 0.1797693134862315907729305190789024733617...
    check_short!(f64::MAX => b"17976931348623157", 309);

    // 10^-307 * 0.2225073858507200889024586876085859887650...
    // 10^-307 * 0.2225073858507201383090232717332404064219...
    // 10^-307 * 0.2225073858507201877155878558578948240788...
    check_short!(f64::MIN_POSITIVE => b"22250738585072014", -307);

    // 10^-323 * 0
    // 10^-323 * 0.4940656458412465441765687928682213723650...
    // 10^-323 * 0.9881312916824930883531375857364427447301...
    let minf64 = ldexp_f64(1.0, -1074);
    check_short!(minf64 => b"5", -323);
}

/// This test ends up running what I can only assume is some corner-ish case
/// of the `exp2` library function, defined in whatever C runtime we're
/// using. In VS 2013 this function apparently had a bug as this test fails
/// when linked, but with VS 2015 the bug appears fixed as the test runs just
/// fine.
///
/// The bug seems to be a difference in return value of `exp2(-1057)`, where
/// in VS 2013 it returns a double with the bit pattern 0x2 and in VS 2015 it
/// returns 0x20000.
///
/// For now just ignore this test entirely on MSVC as it's tested elsewhere
/// anyway and we're not super interested in testing each platform's exp2
/// implementation.
#[test]
#[cfg(not(target_env = "msvc"))]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn f64_fixed_sanity_test() {
    let minf64 = ldexp_f64(1.0, -1074);

    check_fixed_mix!(0.1f64            => b"1000000000000000055511151231257827021182", 0);
    check_fixed_mix!(0.45f64           => b"4500000000000000111022302462515654042363", 0);
    check_fixed_mix!(0.5f64            => b"5=", 0);
    check_fixed_mix!(0.95f64           => b"9499999999999999555910790149937383830547", 0);
    check_fixed_mix!(100.0f64          => b"1=", 3);
    check_fixed_mix!(999.5f64          => b"9995000000000000000000000000000000000000", 3);
    check_fixed_mix!(1.0f64/3.0        => b"3333333333333333148296162562473909929395", 0);
    check_fixed_mix!(3.141592f64       => b"3141592000000000162174274009885266423225", 1);
    check_fixed_mix!(3.141592e17f64    => b"3141592=", 18);
    check_fixed_mix!(1.0e23f64         => b"99999999999999991611392=", 23);
    check_fixed_mix!(f64::MAX          => b"1797693134862315708145274237317043567981", 309);
    check_fixed_mix!(f64::MIN_POSITIVE => b"2225073858507201383090232717332404064219", -307);
    check_fixed_mix!(minf64            => b"4940656458412465441765687928682213723650\
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
                                            7538682506419718265533447265625=", -323);

    // [1], Table 3: Stress Inputs for Converting 53-bit Binary to Decimal, < 1/2 ULP
    check_coef_pow2!(8511030020275656_f64,  -342_f64 => b"9",                       -87);
    check_coef_pow2!(5201988407066741_f64,  -824_f64 => b"46",                     -232);
    check_coef_pow2!(6406892948269899_f64,   237_f64 => b"141",                      88);
    check_coef_pow2!(8431154198732492_f64,    72_f64 => b"3981",                     38);
    check_coef_pow2!(6475049196144587_f64,    99_f64 => b"41040",                    46);
    check_coef_pow2!(8274307542972842_f64,   726_f64 => b"292084",                  235);
    check_coef_pow2!(5381065484265332_f64,  -456_f64 => b"2891946",                -121);
    check_coef_pow2!(6761728585499734_f64, -1057_f64 => b"43787718",               -302);
    check_coef_pow2!(7976538478610756_f64,   376_f64 => b"122770163",               130);
    check_coef_pow2!(5982403858958067_f64,   377_f64 => b"1841552452",              130);
    check_coef_pow2!(5536995190630837_f64,    93_f64 => b"54835744350",              44);
    check_coef_pow2!(7225450889282194_f64,   710_f64 => b"389190181146",            230);
    check_coef_pow2!(7225450889282194_f64,   709_f64 => b"1945950905732",           230);
    check_coef_pow2!(8703372741147379_f64,   117_f64 => b"14460958381605",           52);
    check_coef_pow2!(8944262675275217_f64, -1001_f64 => b"417367747458531",        -285);
    check_coef_pow2!(7459803696087692_f64,  -707_f64 => b"1107950772878888",       -196);
    check_coef_pow2!(6080469016670379_f64,  -381_f64 => b"12345501366327440",       -98);
    check_coef_pow2!(8385515147034757_f64,   721_f64 => b"925031711960365024",      233);
    check_coef_pow2!(7514216811389786_f64,  -828_f64 => b"4198047150284889840",    -233);
    check_coef_pow2!(8397297803260511_f64,  -345_f64 => b"11716315319786511046",    -87);
    check_coef_pow2!(6733459239310543_f64,   202_f64 => b"432810072844612493629",    77);
    check_coef_pow2!(8091450587292794_f64,  -473_f64 => b"3317710118160031081518", -126);

    // [1], Table 4: Stress Inputs for Converting 53-bit Binary to Decimal, > 1/2 ULP
    check_coef_pow2!(6567258882077402_f64,   952_f64 => b"3",                       303);
    check_coef_pow2!(6712731423444934_f64,   535_f64 => b"76",                      177);
    check_coef_pow2!(6712731423444934_f64,   534_f64 => b"378",                     177);
    check_coef_pow2!(5298405411573037_f64,  -957_f64 => b"4350",                   -272);
    check_coef_pow2!(5137311167659507_f64,  -144_f64 => b"23037",                   -27);
    check_coef_pow2!(6722280709661868_f64,   363_f64 => b"126301",                  126);
    check_coef_pow2!(5344436398034927_f64,  -169_f64 => b"7142211",                 -35);
    check_coef_pow2!(8369123604277281_f64,  -853_f64 => b"13934574",               -240);
    check_coef_pow2!(8995822108487663_f64,  -780_f64 => b"141463449",              -218);
    check_coef_pow2!(8942832835564782_f64,  -383_f64 => b"4539277920",              -99);
    check_coef_pow2!(8942832835564782_f64,  -384_f64 => b"22696389598",             -99);
    check_coef_pow2!(8942832835564782_f64,  -385_f64 => b"113481947988",            -99);
    check_coef_pow2!(6965949469487146_f64,  -249_f64 => b"7700366561890",           -59);
    check_coef_pow2!(6965949469487146_f64,  -250_f64 => b"38501832809448",          -59);
    check_coef_pow2!(6965949469487146_f64,  -251_f64 => b"192509164047238",         -59);
    check_coef_pow2!(7487252720986826_f64,   548_f64 => b"6898586531774201",        181);
    check_coef_pow2!(5592117679628511_f64,   164_f64 => b"13076622631878654",        66);
    check_coef_pow2!(8887055249355788_f64,   665_f64 => b"136052020756121240",      217);
    check_coef_pow2!(6994187472632449_f64,   690_f64 => b"3592810217475959676",     224);
    check_coef_pow2!(8797576579012143_f64,   588_f64 => b"89125197712484551899",    193);
    check_coef_pow2!(7363326733505337_f64,   272_f64 => b"558769757362301140950",    98);
    check_coef_pow2!(8549497411294502_f64,  -448_f64 => b"1176257830728540379990", -118);
}

#[test]
fn more_short_sanity_test() {
    let mut buf = [MaybeUninit::new(b'_'); MAX_SIG_DIGITS];
    assert_eq!(
        flt2dec::format_short(
            &Decoded { mant: 99_999_999_999_999_999, minus: 1, plus: 1, exp: 0, inclusive: true },
            &mut buf,
        ),
        ("1".as_bytes(), 18),
    );
    assert_eq!(
        flt2dec::format_short(
            &Decoded { mant: 99_999_999_999_999_999, minus: 1, plus: 1, exp: 0, inclusive: false },
            &mut buf,
        ),
        ("99999999999999999".as_bytes(), 17),
    );
}

fn to_string_with_parts<F>(mut f: F) -> String
where
    F: for<'a> FnMut(&'a mut [MaybeUninit<u8>], &'a mut [MaybeUninit<Part<'a>>]) -> Formatted<'a>,
{
    let mut buf = [MaybeUninit::new(0); 1024];
    let mut parts = [MaybeUninit::new(Part::Zero(0)); 16];
    let formatted = f(&mut buf, &mut parts);
    let mut ret = vec![0; formatted.len()];
    assert_eq!(formatted.write(&mut ret), Some(ret.len()));
    String::from_utf8(ret).unwrap()
}

pub fn to_shortest_str_test<F>(mut f_: F)
where
    F: for<'a> FnMut(&Decoded, &'a mut [MaybeUninit<u8>]) -> (&'a [u8], i16),
{
    fn to_string<T, F>(f: &mut F, v: T, sign: Sign, frac_digits: usize) -> String
    where
        T: DecodableFloat,
        F: for<'a> FnMut(&Decoded, &'a mut [MaybeUninit<u8>]) -> (&'a [u8], i16),
    {
        to_string_with_parts(|buf, parts| {
            to_shortest_str(|d, b| f(d, b), v, sign, frac_digits, buf, parts)
        })
    }

    let f = &mut f_;

    assert_eq!(to_string(f, 0.0, Minus, 0), "0");
    assert_eq!(to_string(f, 0.0, Minus, 0), "0");
    assert_eq!(to_string(f, 0.0, MinusPlus, 0), "+0");
    assert_eq!(to_string(f, -0.0, Minus, 0), "-0");
    assert_eq!(to_string(f, -0.0, MinusPlus, 0), "-0");
    assert_eq!(to_string(f, 0.0, Minus, 1), "0.0");
    assert_eq!(to_string(f, 0.0, Minus, 1), "0.0");
    assert_eq!(to_string(f, 0.0, MinusPlus, 1), "+0.0");
    assert_eq!(to_string(f, -0.0, Minus, 8), "-0.00000000");
    assert_eq!(to_string(f, -0.0, MinusPlus, 8), "-0.00000000");

    assert_eq!(to_string(f, 1.0 / 0.0, Minus, 0), "inf");
    assert_eq!(to_string(f, 1.0 / 0.0, Minus, 0), "inf");
    assert_eq!(to_string(f, 1.0 / 0.0, MinusPlus, 0), "+inf");
    assert_eq!(to_string(f, 0.0 / 0.0, Minus, 0), "NaN");
    assert_eq!(to_string(f, 0.0 / 0.0, Minus, 1), "NaN");
    assert_eq!(to_string(f, 0.0 / 0.0, MinusPlus, 64), "NaN");
    assert_eq!(to_string(f, -1.0 / 0.0, Minus, 0), "-inf");
    assert_eq!(to_string(f, -1.0 / 0.0, Minus, 1), "-inf");
    assert_eq!(to_string(f, -1.0 / 0.0, MinusPlus, 64), "-inf");

    assert_eq!(to_string(f, 3.14, Minus, 0), "3.14");
    assert_eq!(to_string(f, 3.14, Minus, 0), "3.14");
    assert_eq!(to_string(f, 3.14, MinusPlus, 0), "+3.14");
    assert_eq!(to_string(f, -3.14, Minus, 0), "-3.14");
    assert_eq!(to_string(f, -3.14, Minus, 0), "-3.14");
    assert_eq!(to_string(f, -3.14, MinusPlus, 0), "-3.14");
    assert_eq!(to_string(f, 3.14, Minus, 1), "3.14");
    assert_eq!(to_string(f, 3.14, Minus, 2), "3.14");
    assert_eq!(to_string(f, 3.14, MinusPlus, 4), "+3.1400");
    assert_eq!(to_string(f, -3.14, Minus, 8), "-3.14000000");
    assert_eq!(to_string(f, -3.14, Minus, 8), "-3.14000000");
    assert_eq!(to_string(f, -3.14, MinusPlus, 8), "-3.14000000");

    assert_eq!(to_string(f, 7.5e-11, Minus, 0), "0.000000000075");
    assert_eq!(to_string(f, 7.5e-11, Minus, 3), "0.000000000075");
    assert_eq!(to_string(f, 7.5e-11, Minus, 12), "0.000000000075");
    assert_eq!(to_string(f, 7.5e-11, Minus, 13), "0.0000000000750");

    assert_eq!(to_string(f, 1.9971e20, Minus, 0), "199710000000000000000");
    assert_eq!(to_string(f, 1.9971e20, Minus, 1), "199710000000000000000.0");
    assert_eq!(to_string(f, 1.9971e20, Minus, 8), "199710000000000000000.00000000");

    #[cfg(target_has_reliable_f16)]
    {
        // f16
        assert_eq!(to_string(f, f16::MAX, Minus, 0), "65500");
        assert_eq!(to_string(f, f16::MAX, Minus, 1), "65500.0");
        assert_eq!(to_string(f, f16::MAX, Minus, 8), "65500.00000000");

        let minf16 = crate::num::ldexp_f16(1.0, -24);
        assert_eq!(to_string(f, minf16, Minus, 0), "0.00000006");
        assert_eq!(to_string(f, minf16, Minus, 8), "0.00000006");
        assert_eq!(to_string(f, minf16, Minus, 9), "0.000000060");
    }

    {
        // f32
        assert_eq!(to_string(f, f32::MAX, Minus, 0), format!("34028235{:0>31}", ""));
        assert_eq!(to_string(f, f32::MAX, Minus, 1), format!("34028235{:0>31}.0", ""));
        assert_eq!(to_string(f, f32::MAX, Minus, 8), format!("34028235{:0>31}.00000000", ""));

        let minf32 = ldexp_f32(1.0, -149);
        assert_eq!(to_string(f, minf32, Minus, 0), format!("0.{:0>44}1", ""));
        assert_eq!(to_string(f, minf32, Minus, 45), format!("0.{:0>44}1", ""));
        assert_eq!(to_string(f, minf32, Minus, 46), format!("0.{:0>44}10", ""));
    }

    {
        // f64
        assert_eq!(to_string(f, f64::MAX, Minus, 0), format!("17976931348623157{:0>292}", ""));
        assert_eq!(to_string(f, f64::MAX, Minus, 1), format!("17976931348623157{:0>292}.0", ""));
        assert_eq!(
            to_string(f, f64::MAX, Minus, 8),
            format!("17976931348623157{:0>292}.00000000", "")
        );

        let minf64 = ldexp_f64(1.0, -1074);
        assert_eq!(to_string(f, minf64, Minus, 0), format!("0.{:0>323}5", ""));
        assert_eq!(to_string(f, minf64, Minus, 324), format!("0.{:0>323}5", ""));
        assert_eq!(to_string(f, minf64, Minus, 325), format!("0.{:0>323}50", ""));
    }

    if cfg!(miri) {
        // Miri is too slow
        return;
    }

    // very large output
    assert_eq!(to_string(f, 1.1, Minus, 50000), format!("1.1{:0>49999}", ""));
}

pub fn to_shortest_exp_str_test<F>(mut f_: F)
where
    F: for<'a> FnMut(&Decoded, &'a mut [MaybeUninit<u8>]) -> (&'a [u8], i16),
{
    fn to_string<T, F>(f: &mut F, v: T, sign: Sign, exp_bounds: (i16, i16), upper: bool) -> String
    where
        T: DecodableFloat,
        F: for<'a> FnMut(&Decoded, &'a mut [MaybeUninit<u8>]) -> (&'a [u8], i16),
    {
        to_string_with_parts(|buf, parts| {
            to_shortest_exp_str(|d, b| f(d, b), v, sign, exp_bounds, upper, buf, parts)
        })
    }

    let f = &mut f_;

    assert_eq!(to_string(f, 0.0, Minus, (-4, 16), false), "0");
    assert_eq!(to_string(f, 0.0, Minus, (-4, 16), false), "0");
    assert_eq!(to_string(f, 0.0, MinusPlus, (-4, 16), false), "+0");
    assert_eq!(to_string(f, -0.0, Minus, (-4, 16), false), "-0");
    assert_eq!(to_string(f, -0.0, MinusPlus, (-4, 16), false), "-0");
    assert_eq!(to_string(f, 0.0, Minus, (0, 0), true), "0E0");
    assert_eq!(to_string(f, 0.0, Minus, (0, 0), false), "0e0");
    assert_eq!(to_string(f, 0.0, MinusPlus, (5, 9), false), "+0e0");
    assert_eq!(to_string(f, -0.0, Minus, (0, 0), true), "-0E0");
    assert_eq!(to_string(f, -0.0, MinusPlus, (5, 9), false), "-0e0");

    assert_eq!(to_string(f, 1.0 / 0.0, Minus, (-4, 16), false), "inf");
    assert_eq!(to_string(f, 1.0 / 0.0, Minus, (-4, 16), true), "inf");
    assert_eq!(to_string(f, 1.0 / 0.0, MinusPlus, (-4, 16), true), "+inf");
    assert_eq!(to_string(f, 0.0 / 0.0, Minus, (0, 0), false), "NaN");
    assert_eq!(to_string(f, 0.0 / 0.0, Minus, (0, 0), true), "NaN");
    assert_eq!(to_string(f, 0.0 / 0.0, MinusPlus, (5, 9), true), "NaN");
    assert_eq!(to_string(f, -1.0 / 0.0, Minus, (0, 0), false), "-inf");
    assert_eq!(to_string(f, -1.0 / 0.0, Minus, (0, 0), true), "-inf");
    assert_eq!(to_string(f, -1.0 / 0.0, MinusPlus, (5, 9), true), "-inf");

    assert_eq!(to_string(f, 3.14, Minus, (-4, 16), false), "3.14");
    assert_eq!(to_string(f, 3.14, MinusPlus, (-4, 16), false), "+3.14");
    assert_eq!(to_string(f, -3.14, Minus, (-4, 16), false), "-3.14");
    assert_eq!(to_string(f, -3.14, MinusPlus, (-4, 16), false), "-3.14");
    assert_eq!(to_string(f, 3.14, Minus, (0, 0), true), "3.14E0");
    assert_eq!(to_string(f, 3.14, Minus, (0, 0), false), "3.14e0");
    assert_eq!(to_string(f, 3.14, MinusPlus, (5, 9), false), "+3.14e0");
    assert_eq!(to_string(f, -3.14, Minus, (0, 0), true), "-3.14E0");
    assert_eq!(to_string(f, -3.14, Minus, (0, 0), false), "-3.14e0");
    assert_eq!(to_string(f, -3.14, MinusPlus, (5, 9), false), "-3.14e0");

    assert_eq!(to_string(f, 0.1, Minus, (-4, 16), false), "0.1");
    assert_eq!(to_string(f, 0.1, Minus, (-4, 16), false), "0.1");
    assert_eq!(to_string(f, 0.1, MinusPlus, (-4, 16), false), "+0.1");
    assert_eq!(to_string(f, -0.1, Minus, (-4, 16), false), "-0.1");
    assert_eq!(to_string(f, -0.1, MinusPlus, (-4, 16), false), "-0.1");
    assert_eq!(to_string(f, 0.1, Minus, (0, 0), true), "1E-1");
    assert_eq!(to_string(f, 0.1, Minus, (0, 0), false), "1e-1");
    assert_eq!(to_string(f, 0.1, MinusPlus, (5, 9), false), "+1e-1");
    assert_eq!(to_string(f, -0.1, Minus, (0, 0), true), "-1E-1");
    assert_eq!(to_string(f, -0.1, Minus, (0, 0), false), "-1e-1");
    assert_eq!(to_string(f, -0.1, MinusPlus, (5, 9), false), "-1e-1");

    assert_eq!(to_string(f, 7.5e-11, Minus, (-4, 16), false), "7.5e-11");
    assert_eq!(to_string(f, 7.5e-11, Minus, (-11, 10), false), "0.000000000075");
    assert_eq!(to_string(f, 7.5e-11, Minus, (-10, 11), false), "7.5e-11");

    assert_eq!(to_string(f, 1.9971e20, Minus, (-4, 16), false), "1.9971e20");
    assert_eq!(to_string(f, 1.9971e20, Minus, (-20, 21), false), "199710000000000000000");
    assert_eq!(to_string(f, 1.9971e20, Minus, (-21, 20), false), "1.9971e20");

    // the true value of 1.0e23f64 is less than 10^23, but that shouldn't matter here
    assert_eq!(to_string(f, 1.0e23, Minus, (22, 23), false), "1e23");
    assert_eq!(to_string(f, 1.0e23, Minus, (23, 24), false), "100000000000000000000000");
    assert_eq!(to_string(f, 1.0e23, Minus, (24, 25), false), "1e23");

    #[cfg(target_has_reliable_f16)]
    {
        // f16
        assert_eq!(to_string(f, f16::MAX, Minus, (-2, 2), false), "6.55e4");
        assert_eq!(to_string(f, f16::MAX, Minus, (-4, 4), false), "6.55e4");
        assert_eq!(to_string(f, f16::MAX, Minus, (-5, 5), false), "65500");

        let minf16 = crate::num::ldexp_f16(1.0, -24);
        assert_eq!(to_string(f, minf16, Minus, (-2, 2), false), "6e-8");
        assert_eq!(to_string(f, minf16, Minus, (-7, 7), false), "6e-8");
        assert_eq!(to_string(f, minf16, Minus, (-8, 8), false), "0.00000006");
    }

    {
        // f32
        assert_eq!(to_string(f, f32::MAX, Minus, (-4, 16), false), "3.4028235e38");
        assert_eq!(to_string(f, f32::MAX, Minus, (-39, 38), false), "3.4028235e38");
        assert_eq!(to_string(f, f32::MAX, Minus, (-38, 39), false), format!("34028235{:0>31}", ""));

        let minf32 = ldexp_f32(1.0, -149);
        assert_eq!(to_string(f, minf32, Minus, (-4, 16), false), "1e-45");
        assert_eq!(to_string(f, minf32, Minus, (-44, 45), false), "1e-45");
        assert_eq!(to_string(f, minf32, Minus, (-45, 44), false), format!("0.{:0>44}1", ""));
    }

    {
        // f64
        assert_eq!(to_string(f, f64::MAX, Minus, (-4, 16), false), "1.7976931348623157e308");
        assert_eq!(
            to_string(f, f64::MAX, Minus, (-308, 309), false),
            format!("17976931348623157{:0>292}", "")
        );
        assert_eq!(to_string(f, f64::MAX, Minus, (-309, 308), false), "1.7976931348623157e308");

        let minf64 = ldexp_f64(1.0, -1074);
        assert_eq!(to_string(f, minf64, Minus, (-4, 16), false), "5e-324");
        assert_eq!(to_string(f, minf64, Minus, (-324, 323), false), format!("0.{:0>323}5", ""));
        assert_eq!(to_string(f, minf64, Minus, (-323, 324), false), "5e-324");
    }
    assert_eq!(to_string(f, 1.1, Minus, (i16::MIN, i16::MAX), false), "1.1");
}

pub fn to_exact_exp_str_test<F>(mut f_: F)
where
    F: for<'a> FnMut(&Decoded, &'a mut [MaybeUninit<u8>], i16) -> (&'a [u8], i16),
{
    fn to_string<T, F>(f: &mut F, v: T, sign: Sign, ndigits: usize, upper: bool) -> String
    where
        T: DecodableFloat,
        F: for<'a> FnMut(&Decoded, &'a mut [MaybeUninit<u8>], i16) -> (&'a [u8], i16),
    {
        to_string_with_parts(|buf, parts| {
            to_exact_exp_str(|d, b, l| f(d, b, l), v, sign, ndigits, upper, buf, parts)
        })
    }

    let f = &mut f_;

    assert_eq!(to_string(f, 0.0, Minus, 1, true), "0E0");
    assert_eq!(to_string(f, 0.0, Minus, 1, false), "0e0");
    assert_eq!(to_string(f, 0.0, MinusPlus, 1, false), "+0e0");
    assert_eq!(to_string(f, -0.0, Minus, 1, true), "-0E0");
    assert_eq!(to_string(f, -0.0, MinusPlus, 1, false), "-0e0");
    assert_eq!(to_string(f, 0.0, Minus, 2, true), "0.0E0");
    assert_eq!(to_string(f, 0.0, Minus, 2, false), "0.0e0");
    assert_eq!(to_string(f, 0.0, MinusPlus, 2, false), "+0.0e0");
    assert_eq!(to_string(f, -0.0, Minus, 8, false), "-0.0000000e0");
    assert_eq!(to_string(f, -0.0, MinusPlus, 8, false), "-0.0000000e0");

    assert_eq!(to_string(f, 1.0 / 0.0, Minus, 1, false), "inf");
    assert_eq!(to_string(f, 1.0 / 0.0, Minus, 1, true), "inf");
    assert_eq!(to_string(f, 1.0 / 0.0, MinusPlus, 1, true), "+inf");
    assert_eq!(to_string(f, 0.0 / 0.0, Minus, 8, false), "NaN");
    assert_eq!(to_string(f, 0.0 / 0.0, Minus, 8, true), "NaN");
    assert_eq!(to_string(f, 0.0 / 0.0, MinusPlus, 8, true), "NaN");
    assert_eq!(to_string(f, -1.0 / 0.0, Minus, 64, false), "-inf");
    assert_eq!(to_string(f, -1.0 / 0.0, Minus, 64, true), "-inf");
    assert_eq!(to_string(f, -1.0 / 0.0, MinusPlus, 64, true), "-inf");

    assert_eq!(to_string(f, 3.14, Minus, 1, true), "3E0");
    assert_eq!(to_string(f, 3.14, Minus, 1, false), "3e0");
    assert_eq!(to_string(f, 3.14, MinusPlus, 1, false), "+3e0");
    assert_eq!(to_string(f, -3.14, Minus, 2, true), "-3.1E0");
    assert_eq!(to_string(f, -3.14, Minus, 2, false), "-3.1e0");
    assert_eq!(to_string(f, -3.14, MinusPlus, 2, false), "-3.1e0");
    assert_eq!(to_string(f, 3.14, Minus, 3, true), "3.14E0");
    assert_eq!(to_string(f, 3.14, Minus, 3, false), "3.14e0");
    assert_eq!(to_string(f, 3.14, MinusPlus, 3, false), "+3.14e0");
    assert_eq!(to_string(f, -3.14, Minus, 4, true), "-3.140E0");
    assert_eq!(to_string(f, -3.14, Minus, 4, false), "-3.140e0");
    assert_eq!(to_string(f, -3.14, MinusPlus, 4, false), "-3.140e0");

    assert_eq!(to_string(f, 0.195, Minus, 1, false), "2e-1");
    assert_eq!(to_string(f, 0.195, Minus, 1, true), "2E-1");
    assert_eq!(to_string(f, 0.195, MinusPlus, 1, true), "+2E-1");
    assert_eq!(to_string(f, -0.195, Minus, 2, false), "-2.0e-1");
    assert_eq!(to_string(f, -0.195, Minus, 2, true), "-2.0E-1");
    assert_eq!(to_string(f, -0.195, MinusPlus, 2, true), "-2.0E-1");
    assert_eq!(to_string(f, 0.195, Minus, 3, false), "1.95e-1");
    assert_eq!(to_string(f, 0.195, Minus, 3, true), "1.95E-1");
    assert_eq!(to_string(f, 0.195, MinusPlus, 3, true), "+1.95E-1");
    assert_eq!(to_string(f, -0.195, Minus, 4, false), "-1.950e-1");
    assert_eq!(to_string(f, -0.195, Minus, 4, true), "-1.950E-1");
    assert_eq!(to_string(f, -0.195, MinusPlus, 4, true), "-1.950E-1");

    assert_eq!(to_string(f, 9.5, Minus, 1, false), "1e1");
    assert_eq!(to_string(f, 9.5, Minus, 2, false), "9.5e0");
    assert_eq!(to_string(f, 9.5, Minus, 3, false), "9.50e0");
    assert_eq!(to_string(f, 9.5, Minus, 30, false), "9.50000000000000000000000000000e0");

    assert_eq!(to_string(f, 1.0e25, Minus, 1, false), "1e25");
    assert_eq!(to_string(f, 1.0e25, Minus, 2, false), "1.0e25");
    assert_eq!(to_string(f, 1.0e25, Minus, 15, false), "1.00000000000000e25");
    assert_eq!(to_string(f, 1.0e25, Minus, 16, false), "1.000000000000000e25");
    assert_eq!(to_string(f, 1.0e25, Minus, 17, false), "1.0000000000000001e25");
    assert_eq!(to_string(f, 1.0e25, Minus, 18, false), "1.00000000000000009e25");
    assert_eq!(to_string(f, 1.0e25, Minus, 19, false), "1.000000000000000091e25");
    assert_eq!(to_string(f, 1.0e25, Minus, 20, false), "1.0000000000000000906e25");
    assert_eq!(to_string(f, 1.0e25, Minus, 21, false), "1.00000000000000009060e25");
    assert_eq!(to_string(f, 1.0e25, Minus, 22, false), "1.000000000000000090597e25");
    assert_eq!(to_string(f, 1.0e25, Minus, 23, false), "1.0000000000000000905970e25");
    assert_eq!(to_string(f, 1.0e25, Minus, 24, false), "1.00000000000000009059697e25");
    assert_eq!(to_string(f, 1.0e25, Minus, 25, false), "1.000000000000000090596966e25");
    assert_eq!(to_string(f, 1.0e25, Minus, 26, false), "1.0000000000000000905969664e25");
    assert_eq!(to_string(f, 1.0e25, Minus, 27, false), "1.00000000000000009059696640e25");
    assert_eq!(to_string(f, 1.0e25, Minus, 30, false), "1.00000000000000009059696640000e25");

    assert_eq!(to_string(f, 1.0e-6, Minus, 1, false), "1e-6");
    assert_eq!(to_string(f, 1.0e-6, Minus, 2, false), "1.0e-6");
    assert_eq!(to_string(f, 1.0e-6, Minus, 16, false), "1.000000000000000e-6");
    assert_eq!(to_string(f, 1.0e-6, Minus, 17, false), "9.9999999999999995e-7");
    assert_eq!(to_string(f, 1.0e-6, Minus, 18, false), "9.99999999999999955e-7");
    assert_eq!(to_string(f, 1.0e-6, Minus, 19, false), "9.999999999999999547e-7");
    assert_eq!(to_string(f, 1.0e-6, Minus, 20, false), "9.9999999999999995475e-7");
    assert_eq!(to_string(f, 1.0e-6, Minus, 30, false), "9.99999999999999954748111825886e-7");
    assert_eq!(
        to_string(f, 1.0e-6, Minus, 40, false),
        "9.999999999999999547481118258862586856139e-7"
    );
    assert_eq!(
        to_string(f, 1.0e-6, Minus, 50, false),
        "9.9999999999999995474811182588625868561393872369081e-7"
    );
    assert_eq!(
        to_string(f, 1.0e-6, Minus, 60, false),
        "9.99999999999999954748111825886258685613938723690807819366455e-7"
    );
    assert_eq!(
        to_string(f, 1.0e-6, Minus, 70, false),
        "9.999999999999999547481118258862586856139387236908078193664550781250000e-7"
    );

    #[cfg(target_has_reliable_f16)]
    {
        assert_eq!(to_string(f, f16::MAX, Minus, 1, false), "7e4");
        assert_eq!(to_string(f, f16::MAX, Minus, 2, false), "6.6e4");
        assert_eq!(to_string(f, f16::MAX, Minus, 4, false), "6.550e4");
        assert_eq!(to_string(f, f16::MAX, Minus, 5, false), "6.5504e4");
        assert_eq!(to_string(f, f16::MAX, Minus, 6, false), "6.55040e4");
        assert_eq!(to_string(f, f16::MAX, Minus, 16, false), "6.550400000000000e4");

        let minf16 = crate::num::ldexp_f16(1.0, -24);
        assert_eq!(to_string(f, minf16, Minus, 1, false), "6e-8");
        assert_eq!(to_string(f, minf16, Minus, 2, false), "6.0e-8");
        assert_eq!(to_string(f, minf16, Minus, 4, false), "5.960e-8");
        assert_eq!(to_string(f, minf16, Minus, 8, false), "5.9604645e-8");
        assert_eq!(to_string(f, minf16, Minus, 16, false), "5.960464477539062e-8");
        assert_eq!(to_string(f, minf16, Minus, 17, false), "5.9604644775390625e-8");
        assert_eq!(to_string(f, minf16, Minus, 18, false), "5.96046447753906250e-8");
        assert_eq!(to_string(f, minf16, Minus, 24, false), "5.96046447753906250000000e-8");
    }

    assert_eq!(to_string(f, f32::MAX, Minus, 1, false), "3e38");
    assert_eq!(to_string(f, f32::MAX, Minus, 2, false), "3.4e38");
    assert_eq!(to_string(f, f32::MAX, Minus, 4, false), "3.403e38");
    assert_eq!(to_string(f, f32::MAX, Minus, 8, false), "3.4028235e38");
    assert_eq!(to_string(f, f32::MAX, Minus, 16, false), "3.402823466385289e38");
    assert_eq!(to_string(f, f32::MAX, Minus, 32, false), "3.4028234663852885981170418348452e38");
    assert_eq!(
        to_string(f, f32::MAX, Minus, 64, false),
        "3.402823466385288598117041834845169254400000000000000000000000000e38"
    );

    let minf32 = ldexp_f32(1.0, -149);
    assert_eq!(to_string(f, minf32, Minus, 1, false), "1e-45");
    assert_eq!(to_string(f, minf32, Minus, 2, false), "1.4e-45");
    assert_eq!(to_string(f, minf32, Minus, 4, false), "1.401e-45");
    assert_eq!(to_string(f, minf32, Minus, 8, false), "1.4012985e-45");
    assert_eq!(to_string(f, minf32, Minus, 16, false), "1.401298464324817e-45");
    assert_eq!(to_string(f, minf32, Minus, 32, false), "1.4012984643248170709237295832899e-45");
    assert_eq!(
        to_string(f, minf32, Minus, 64, false),
        "1.401298464324817070923729583289916131280261941876515771757068284e-45"
    );
    assert_eq!(
        to_string(f, minf32, Minus, 128, false),
        "1.401298464324817070923729583289916131280261941876515771757068283\
                 8897910826858606014866381883621215820312500000000000000000000000e-45"
    );

    if cfg!(miri) {
        // Miri is too slow
        return;
    }

    assert_eq!(to_string(f, f64::MAX, Minus, 1, false), "2e308");
    assert_eq!(to_string(f, f64::MAX, Minus, 2, false), "1.8e308");
    assert_eq!(to_string(f, f64::MAX, Minus, 4, false), "1.798e308");
    assert_eq!(to_string(f, f64::MAX, Minus, 8, false), "1.7976931e308");
    assert_eq!(to_string(f, f64::MAX, Minus, 16, false), "1.797693134862316e308");
    assert_eq!(to_string(f, f64::MAX, Minus, 32, false), "1.7976931348623157081452742373170e308");
    assert_eq!(
        to_string(f, f64::MAX, Minus, 64, false),
        "1.797693134862315708145274237317043567980705675258449965989174768e308"
    );
    assert_eq!(
        to_string(f, f64::MAX, Minus, 128, false),
        "1.797693134862315708145274237317043567980705675258449965989174768\
                 0315726078002853876058955863276687817154045895351438246423432133e308"
    );
    assert_eq!(
        to_string(f, f64::MAX, Minus, 256, false),
        "1.797693134862315708145274237317043567980705675258449965989174768\
                 0315726078002853876058955863276687817154045895351438246423432132\
                 6889464182768467546703537516986049910576551282076245490090389328\
                 9440758685084551339423045832369032229481658085593321233482747978e308"
    );
    assert_eq!(
        to_string(f, f64::MAX, Minus, 512, false),
        "1.797693134862315708145274237317043567980705675258449965989174768\
                 0315726078002853876058955863276687817154045895351438246423432132\
                 6889464182768467546703537516986049910576551282076245490090389328\
                 9440758685084551339423045832369032229481658085593321233482747978\
                 2620414472316873817718091929988125040402618412485836800000000000\
                 0000000000000000000000000000000000000000000000000000000000000000\
                 0000000000000000000000000000000000000000000000000000000000000000\
                 0000000000000000000000000000000000000000000000000000000000000000e308"
    );

    // okay, this is becoming tough. fortunately for us, this is almost the worst case.
    let minf64 = ldexp_f64(1.0, -1074);
    assert_eq!(to_string(f, minf64, Minus, 1, false), "5e-324");
    assert_eq!(to_string(f, minf64, Minus, 2, false), "4.9e-324");
    assert_eq!(to_string(f, minf64, Minus, 4, false), "4.941e-324");
    assert_eq!(to_string(f, minf64, Minus, 8, false), "4.9406565e-324");
    assert_eq!(to_string(f, minf64, Minus, 16, false), "4.940656458412465e-324");
    assert_eq!(to_string(f, minf64, Minus, 32, false), "4.9406564584124654417656879286822e-324");
    assert_eq!(
        to_string(f, minf64, Minus, 64, false),
        "4.940656458412465441765687928682213723650598026143247644255856825e-324"
    );
    assert_eq!(
        to_string(f, minf64, Minus, 128, false),
        "4.940656458412465441765687928682213723650598026143247644255856825\
                 0067550727020875186529983636163599237979656469544571773092665671e-324"
    );
    assert_eq!(
        to_string(f, minf64, Minus, 256, false),
        "4.940656458412465441765687928682213723650598026143247644255856825\
                 0067550727020875186529983636163599237979656469544571773092665671\
                 0355939796398774796010781878126300713190311404527845817167848982\
                 1036887186360569987307230500063874091535649843873124733972731696e-324"
    );
    assert_eq!(
        to_string(f, minf64, Minus, 512, false),
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
        to_string(f, minf64, Minus, 1024, false),
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
    assert_eq!(to_string(f, 0.0, Minus, 50000, false), format!("0.{:0>49999}e0", ""));
    assert_eq!(to_string(f, 1.0e1, Minus, 50000, false), format!("1.{:0>49999}e1", ""));
    assert_eq!(to_string(f, 1.0e0, Minus, 50000, false), format!("1.{:0>49999}e0", ""));
    assert_eq!(
        to_string(f, 1.0e-1, Minus, 50000, false),
        format!(
            "1.000000000000000055511151231257827021181583404541015625{:0>49945}\
                        e-1",
            ""
        )
    );
    assert_eq!(
        to_string(f, 1.0e-20, Minus, 50000, false),
        format!(
            "9.999999999999999451532714542095716517295037027873924471077157760\
                         66783064379706047475337982177734375{:0>49901}e-21",
            ""
        )
    );
}

pub fn to_exact_fixed_str_test<F>(mut f_: F)
where
    F: for<'a> FnMut(&Decoded, &'a mut [MaybeUninit<u8>], i16) -> (&'a [u8], i16),
{
    fn to_string<T, F>(f: &mut F, v: T, sign: Sign, frac_digits: usize) -> String
    where
        T: DecodableFloat,
        F: for<'a> FnMut(&Decoded, &'a mut [MaybeUninit<u8>], i16) -> (&'a [u8], i16),
    {
        to_string_with_parts(|buf, parts| {
            to_exact_fixed_str(|d, b, l| f(d, b, l), v, sign, frac_digits, buf, parts)
        })
    }

    let f = &mut f_;

    assert_eq!(to_string(f, 0.0, Minus, 0), "0");
    assert_eq!(to_string(f, 0.0, MinusPlus, 0), "+0");
    assert_eq!(to_string(f, -0.0, Minus, 0), "-0");
    assert_eq!(to_string(f, -0.0, MinusPlus, 0), "-0");
    assert_eq!(to_string(f, 0.0, Minus, 1), "0.0");
    assert_eq!(to_string(f, 0.0, MinusPlus, 1), "+0.0");
    assert_eq!(to_string(f, -0.0, Minus, 8), "-0.00000000");
    assert_eq!(to_string(f, -0.0, MinusPlus, 8), "-0.00000000");

    assert_eq!(to_string(f, 1.0 / 0.0, Minus, 0), "inf");
    assert_eq!(to_string(f, 1.0 / 0.0, Minus, 1), "inf");
    assert_eq!(to_string(f, 1.0 / 0.0, MinusPlus, 64), "+inf");
    assert_eq!(to_string(f, 0.0 / 0.0, Minus, 0), "NaN");
    assert_eq!(to_string(f, 0.0 / 0.0, Minus, 1), "NaN");
    assert_eq!(to_string(f, 0.0 / 0.0, MinusPlus, 64), "NaN");
    assert_eq!(to_string(f, -1.0 / 0.0, Minus, 0), "-inf");
    assert_eq!(to_string(f, -1.0 / 0.0, Minus, 1), "-inf");
    assert_eq!(to_string(f, -1.0 / 0.0, MinusPlus, 64), "-inf");

    assert_eq!(to_string(f, 3.14, Minus, 0), "3");
    assert_eq!(to_string(f, 3.14, Minus, 0), "3");
    assert_eq!(to_string(f, 3.14, MinusPlus, 0), "+3");
    assert_eq!(to_string(f, -3.14, Minus, 0), "-3");
    assert_eq!(to_string(f, -3.14, Minus, 0), "-3");
    assert_eq!(to_string(f, -3.14, MinusPlus, 0), "-3");
    assert_eq!(to_string(f, 3.14, Minus, 1), "3.1");
    assert_eq!(to_string(f, 3.14, Minus, 2), "3.14");
    assert_eq!(to_string(f, 3.14, MinusPlus, 4), "+3.1400");
    assert_eq!(to_string(f, -3.14, Minus, 8), "-3.14000000");
    assert_eq!(to_string(f, -3.14, Minus, 8), "-3.14000000");
    assert_eq!(to_string(f, -3.14, MinusPlus, 8), "-3.14000000");

    assert_eq!(to_string(f, 0.195, Minus, 0), "0");
    assert_eq!(to_string(f, 0.195, MinusPlus, 0), "+0");
    assert_eq!(to_string(f, -0.195, Minus, 0), "-0");
    assert_eq!(to_string(f, -0.195, Minus, 0), "-0");
    assert_eq!(to_string(f, -0.195, MinusPlus, 0), "-0");
    assert_eq!(to_string(f, 0.195, Minus, 1), "0.2");
    assert_eq!(to_string(f, 0.195, Minus, 2), "0.20");
    assert_eq!(to_string(f, 0.195, MinusPlus, 4), "+0.1950");
    assert_eq!(to_string(f, -0.195, Minus, 5), "-0.19500");
    assert_eq!(to_string(f, -0.195, Minus, 6), "-0.195000");
    assert_eq!(to_string(f, -0.195, MinusPlus, 8), "-0.19500000");

    assert_eq!(to_string(f, 999.5, Minus, 0), "1000");
    assert_eq!(to_string(f, 999.5, Minus, 1), "999.5");
    assert_eq!(to_string(f, 999.5, Minus, 2), "999.50");
    assert_eq!(to_string(f, 999.5, Minus, 3), "999.500");
    assert_eq!(to_string(f, 999.5, Minus, 30), "999.500000000000000000000000000000");

    assert_eq!(to_string(f, 0.5, Minus, 0), "0");
    assert_eq!(to_string(f, 0.5, Minus, 1), "0.5");
    assert_eq!(to_string(f, 0.5, Minus, 2), "0.50");
    assert_eq!(to_string(f, 0.5, Minus, 3), "0.500");

    assert_eq!(to_string(f, 0.95, Minus, 0), "1");
    assert_eq!(to_string(f, 0.95, Minus, 1), "0.9"); // because it really is less than 0.95
    assert_eq!(to_string(f, 0.95, Minus, 2), "0.95");
    assert_eq!(to_string(f, 0.95, Minus, 3), "0.950");
    assert_eq!(to_string(f, 0.95, Minus, 10), "0.9500000000");
    assert_eq!(to_string(f, 0.95, Minus, 30), "0.949999999999999955591079014994");

    assert_eq!(to_string(f, 0.095, Minus, 0), "0");
    assert_eq!(to_string(f, 0.095, Minus, 1), "0.1");
    assert_eq!(to_string(f, 0.095, Minus, 2), "0.10");
    assert_eq!(to_string(f, 0.095, Minus, 3), "0.095");
    assert_eq!(to_string(f, 0.095, Minus, 4), "0.0950");
    assert_eq!(to_string(f, 0.095, Minus, 10), "0.0950000000");
    assert_eq!(to_string(f, 0.095, Minus, 30), "0.095000000000000001110223024625");

    assert_eq!(to_string(f, 0.0095, Minus, 0), "0");
    assert_eq!(to_string(f, 0.0095, Minus, 1), "0.0");
    assert_eq!(to_string(f, 0.0095, Minus, 2), "0.01");
    assert_eq!(to_string(f, 0.0095, Minus, 3), "0.009"); // really is less than 0.0095
    assert_eq!(to_string(f, 0.0095, Minus, 4), "0.0095");
    assert_eq!(to_string(f, 0.0095, Minus, 5), "0.00950");
    assert_eq!(to_string(f, 0.0095, Minus, 10), "0.0095000000");
    assert_eq!(to_string(f, 0.0095, Minus, 30), "0.009499999999999999764077607267");

    assert_eq!(to_string(f, 7.5e-11, Minus, 0), "0");
    assert_eq!(to_string(f, 7.5e-11, Minus, 3), "0.000");
    assert_eq!(to_string(f, 7.5e-11, Minus, 10), "0.0000000001");
    assert_eq!(to_string(f, 7.5e-11, Minus, 11), "0.00000000007"); // ditto
    assert_eq!(to_string(f, 7.5e-11, Minus, 12), "0.000000000075");
    assert_eq!(to_string(f, 7.5e-11, Minus, 13), "0.0000000000750");
    assert_eq!(to_string(f, 7.5e-11, Minus, 20), "0.00000000007500000000");
    assert_eq!(to_string(f, 7.5e-11, Minus, 30), "0.000000000074999999999999999501");

    assert_eq!(to_string(f, 1.0e25, Minus, 0), "10000000000000000905969664");
    assert_eq!(to_string(f, 1.0e25, Minus, 1), "10000000000000000905969664.0");
    assert_eq!(to_string(f, 1.0e25, Minus, 3), "10000000000000000905969664.000");

    assert_eq!(to_string(f, 1.0e-6, Minus, 0), "0");
    assert_eq!(to_string(f, 1.0e-6, Minus, 3), "0.000");
    assert_eq!(to_string(f, 1.0e-6, Minus, 6), "0.000001");
    assert_eq!(to_string(f, 1.0e-6, Minus, 9), "0.000001000");
    assert_eq!(to_string(f, 1.0e-6, Minus, 12), "0.000001000000");
    assert_eq!(to_string(f, 1.0e-6, Minus, 22), "0.0000010000000000000000");
    assert_eq!(to_string(f, 1.0e-6, Minus, 23), "0.00000099999999999999995");
    assert_eq!(to_string(f, 1.0e-6, Minus, 24), "0.000000999999999999999955");
    assert_eq!(to_string(f, 1.0e-6, Minus, 25), "0.0000009999999999999999547");
    assert_eq!(to_string(f, 1.0e-6, Minus, 35), "0.00000099999999999999995474811182589");
    assert_eq!(to_string(f, 1.0e-6, Minus, 45), "0.000000999999999999999954748111825886258685614");
    assert_eq!(
        to_string(f, 1.0e-6, Minus, 55),
        "0.0000009999999999999999547481118258862586856139387236908"
    );
    assert_eq!(
        to_string(f, 1.0e-6, Minus, 65),
        "0.00000099999999999999995474811182588625868561393872369080781936646"
    );
    assert_eq!(
        to_string(f, 1.0e-6, Minus, 75),
        "0.000000999999999999999954748111825886258685613938723690807819366455078125000"
    );

    #[cfg(target_has_reliable_f16)]
    {
        assert_eq!(to_string(f, f16::MAX, Minus, 0), "65504");
        assert_eq!(to_string(f, f16::MAX, Minus, 1), "65504.0");
        assert_eq!(to_string(f, f16::MAX, Minus, 2), "65504.00");
    }

    assert_eq!(to_string(f, f32::MAX, Minus, 0), "340282346638528859811704183484516925440");
    assert_eq!(to_string(f, f32::MAX, Minus, 1), "340282346638528859811704183484516925440.0");
    assert_eq!(to_string(f, f32::MAX, Minus, 2), "340282346638528859811704183484516925440.00");

    if cfg!(miri) {
        // Miri is too slow
        return;
    }

    #[cfg(target_has_reliable_f16)]
    {
        let minf16 = crate::num::ldexp_f16(1.0, -24);
        assert_eq!(to_string(f, minf16, Minus, 0), "0");
        assert_eq!(to_string(f, minf16, Minus, 1), "0.0");
        assert_eq!(to_string(f, minf16, Minus, 2), "0.00");
        assert_eq!(to_string(f, minf16, Minus, 4), "0.0000");
        assert_eq!(to_string(f, minf16, Minus, 8), "0.00000006");
        assert_eq!(to_string(f, minf16, Minus, 10), "0.0000000596");
        assert_eq!(to_string(f, minf16, Minus, 15), "0.000000059604645");
        assert_eq!(to_string(f, minf16, Minus, 20), "0.00000005960464477539");
        assert_eq!(to_string(f, minf16, Minus, 24), "0.000000059604644775390625");
        assert_eq!(to_string(f, minf16, Minus, 32), "0.00000005960464477539062500000000");
    }

    let minf32 = ldexp_f32(1.0, -149);
    assert_eq!(to_string(f, minf32, Minus, 0), "0");
    assert_eq!(to_string(f, minf32, Minus, 1), "0.0");
    assert_eq!(to_string(f, minf32, Minus, 2), "0.00");
    assert_eq!(to_string(f, minf32, Minus, 4), "0.0000");
    assert_eq!(to_string(f, minf32, Minus, 8), "0.00000000");
    assert_eq!(to_string(f, minf32, Minus, 16), "0.0000000000000000");
    assert_eq!(to_string(f, minf32, Minus, 32), "0.00000000000000000000000000000000");
    assert_eq!(
        to_string(f, minf32, Minus, 64),
        "0.0000000000000000000000000000000000000000000014012984643248170709"
    );
    assert_eq!(
        to_string(f, minf32, Minus, 128),
        "0.0000000000000000000000000000000000000000000014012984643248170709\
                  2372958328991613128026194187651577175706828388979108268586060149"
    );
    assert_eq!(
        to_string(f, minf32, Minus, 256),
        "0.0000000000000000000000000000000000000000000014012984643248170709\
                  2372958328991613128026194187651577175706828388979108268586060148\
                  6638188362121582031250000000000000000000000000000000000000000000\
                  0000000000000000000000000000000000000000000000000000000000000000"
    );

    assert_eq!(
        to_string(f, f64::MAX, Minus, 0),
        "1797693134862315708145274237317043567980705675258449965989174768\
                0315726078002853876058955863276687817154045895351438246423432132\
                6889464182768467546703537516986049910576551282076245490090389328\
                9440758685084551339423045832369032229481658085593321233482747978\
                26204144723168738177180919299881250404026184124858368"
    );
    assert_eq!(
        to_string(f, f64::MAX, Minus, 10),
        "1797693134862315708145274237317043567980705675258449965989174768\
                0315726078002853876058955863276687817154045895351438246423432132\
                6889464182768467546703537516986049910576551282076245490090389328\
                9440758685084551339423045832369032229481658085593321233482747978\
                26204144723168738177180919299881250404026184124858368.0000000000"
    );

    let minf64 = ldexp_f64(1.0, -1074);
    assert_eq!(to_string(f, minf64, Minus, 0), "0");
    assert_eq!(to_string(f, minf64, Minus, 1), "0.0");
    assert_eq!(to_string(f, minf64, Minus, 10), "0.0000000000");
    assert_eq!(
        to_string(f, minf64, Minus, 100),
        "0.0000000000000000000000000000000000000000000000000000000000000000\
                  000000000000000000000000000000000000"
    );
    assert_eq!(
        to_string(f, minf64, Minus, 1000),
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
    assert_eq!(to_string(f, 0.0, Minus, 50000), format!("0.{:0>50000}", ""));
    assert_eq!(to_string(f, 1.0e1, Minus, 50000), format!("10.{:0>50000}", ""));
    assert_eq!(to_string(f, 1.0e0, Minus, 50000), format!("1.{:0>50000}", ""));
    assert_eq!(
        to_string(f, 1.0e-1, Minus, 50000),
        format!("0.1000000000000000055511151231257827021181583404541015625{:0>49945}", "")
    );
    assert_eq!(
        to_string(f, 1.0e-20, Minus, 50000),
        format!(
            "0.0000000000000000000099999999999999994515327145420957165172950370\
                          2787392447107715776066783064379706047475337982177734375{:0>49881}",
            ""
        )
    );
}
