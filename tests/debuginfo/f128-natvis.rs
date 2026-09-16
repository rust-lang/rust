//@ compile-flags: -g
//@ only-msvc

// This tests the `f128` Natvis visualiser.
//@ cdb-command:g
//@ cdb-command:dx v0_0
//@ cdb-check:v0_0             : 0x0p+0 [Type: f128]
//@ cdb-check:bits             : 0x00000000000000000000000000000000
//@ cdb-command:dx neg_0_0
//@ cdb-check:neg_0_0          : -0x0p+0 [Type: f128]
//@ cdb-check:bits             : 0x80000000000000000000000000000000
//@ cdb-command:dx v1_0
//@ cdb-check:v1_0             : 0x1p+0 [Type: f128]
//@ cdb-check:bits             : 0x3fff0000000000000000000000000000
//@ cdb-command:dx v1_5
//@ cdb-check:v1_5             : 0x1.8p+0 [Type: f128]
//@ cdb-check:bits             : 0x3fff8000000000000000000000000000
//@ cdb-command:dx v72_3
//@ cdb-check:v72_3            : 0x1.2133333333333333333333333333p+6 [Type: f128]
//@ cdb-check:bits             : 0x40052133333333333333333333333333
//@ cdb-command:dx neg_0_126
//@ cdb-check:neg_0_126        : -0x1.020c49ba5e353f7ced916872b021p-3 [Type: f128]
//@ cdb-check:bits             : 0xbffc020c49ba5e353f7ced916872b021
//@ cdb-command:dx v0_00003
//@ cdb-check:v0_00003         : 0x1.f75104d551d68c692f6e82949a56p-16 [Type: f128]
//@ cdb-check:bits             : 0x3feff75104d551d68c692f6e82949a56
//@ cdb-command:dx neg_0_00004
//@ cdb-check:neg_0_00004      : -0x1.4f8b588e368f08461f9f01b866e4p-15 [Type: f128]
//@ cdb-check:bits             : 0xbff04f8b588e368f08461f9f01b866e4
//@ cdb-command:dx very_small
//@ cdb-check:very_small       : 0x1p-16494 [Type: f128]
//@ cdb-check:bits             : 0x00000000000000000000000000000001
//@ cdb-command:dx not_quite_as_small
//@ cdb-check:not_quite_as_small : 0x1.8p-16385 [Type: f128]
//@ cdb-check:bits             : 0x00003000000000000000000000000000
//@ cdb-command:dx smallest_pos_normal
//@ cdb-check:smallest_pos_normal : 0x1p-16382 [Type: f128]
//@ cdb-check:bits             : 0x00010000000000000000000000000000
//@ cdb-command:dx smallest_subnormal
//@ cdb-check:smallest_subnormal : -0x1.fffffffffffffffffffffffffffep-16383 [Type: f128]
//@ cdb-check:bits             : 0x8000ffffffffffffffffffffffffffff
//@ cdb-command:dx just_above
//@ cdb-check:just_above       : -0x1.ffffffffffffffffffffffffff8p-1 [Type: f128]
//@ cdb-check:bits             : 0xbffeffffffffffffffffffffffffff80
//@ cdb-command:dx max
//@ cdb-check:max              : 0x1.ffffffffffffffffffffffffffffp+16383 [Type: f128]
//@ cdb-check:bits             : 0x7ffeffffffffffffffffffffffffffff
//@ cdb-command:dx min
//@ cdb-check:min              : -0x1.ffffffffffffffffffffffffffffp+16383 [Type: f128]
//@ cdb-check:bits             : 0xfffeffffffffffffffffffffffffffff
//@ cdb-command:dx inf
//@ cdb-check:inf              : inf [Type: f128]
//@ cdb-check:bits             : 0x7fff0000000000000000000000000000
//@ cdb-command:dx neg_inf
//@ cdb-check:neg_inf          : -inf [Type: f128]
//@ cdb-check:bits             : 0xffff0000000000000000000000000000
//@ cdb-command:dx nan
//@ cdb-check:nan              : NaN [Type: f128]
//@ cdb-check:bits             : 0x7fff8000000000000000000000000000
//@ cdb-command:dx other_nan
//@ cdb-check:other_nan        : NaN [Type: f128]
//@ cdb-check:bits             : 0xffff123456789abcdef123456789abcd

#![feature(f128)]

fn main() {
    let v0_0 = 0.0_f128;
    let neg_0_0 = -0.0_f128;
    let v1_0 = 1.0_f128;
    let v1_5 = 1.5_f128;
    let v72_3 = 72.3_f128;
    let neg_0_126 = -0.126_f128;
    let v0_00003 = 0.00003_f128;
    let neg_0_00004 = -0.00004_f128;
    let very_small = 0.0_f128.next_up();
    let not_quite_as_small = const { f128::MIN_POSITIVE / 8.0 + f128::MIN_POSITIVE / 16.0 };
    let smallest_pos_normal = f128::MIN_POSITIVE;
    let smallest_subnormal = (-f128::MIN_POSITIVE).next_up();
    let just_above = const { -1.0 + f128::EPSILON * 64.0 };
    let max = f128::MAX;
    let min = f128::MIN;
    let inf = f128::INFINITY;
    let neg_inf = f128::NEG_INFINITY;
    let nan = f128::NAN;
    let other_nan = f128::from_bits(0xffff_1234_5678_9abc_def1_2345_6789_abcd);

    _zzz(); // #break
}

fn _zzz() {
    ()
}
