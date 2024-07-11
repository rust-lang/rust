//@ compile-flags: -g
//@ only-msvc

// This tests the `f16` Natvis visualiser.
// cdb-command:g
// cdb-command:dx v0_0
// cdb-check:v0_0             : 0.000000 [Type: f16]
// cdb-command:dx neg_0_0
// cdb-check:neg_0_0          : -0.000000 [Type: f16]
// cdb-command:dx v1_0
// cdb-check:v1_0             : 1.000000 [Type: f16]
// cdb-command:dx v1_5
// cdb-check:v1_5             : 1.500000 [Type: f16]
// cdb-command:dx v72_3
// cdb-check:v72_3            : 72.312500 [Type: f16]
// cdb-command:dx neg_0_126
// cdb-check:neg_0_126        : -0.125977 [Type: f16]
// cdb-command:dx v0_00003
// cdb-check:v0_00003         : 0.000030 [Type: f16]
// cdb-command:dx neg_0_00004
// cdb-check:neg_0_00004      : -0.000040 [Type: f16]
// cdb-command:dx max
// cdb-check:max              : 65504.000000 [Type: f16]
// cdb-command:dx min
// cdb-check:min              : -65504.000000 [Type: f16]
// cdb-command:dx inf
// cdb-check:inf              : inf [Type: f16]
// cdb-command:dx neg_inf
// cdb-check:neg_inf          : -inf [Type: f16]
// cdb-command:dx nan
// cdb-check:nan              : NaN [Type: f16]
// cdb-command:dx other_nan
// cdb-check:other_nan        : NaN [Type: f16]

#![feature(f16)]

fn main() {
    let v0_0 = 0.0_f16;
    let neg_0_0 = -0.0_f16;
    let v1_0 = 1.0_f16;
    let v1_5 = 1.5_f16;
    let v72_3 = 72.3_f16;
    let neg_0_126 = -0.126_f16;
    let v0_00003 = 0.00003_f16;
    let neg_0_00004 = -0.00004_f16;
    let max = f16::MAX;
    let min = f16::MIN;
    let inf = f16::INFINITY;
    let neg_inf = f16::NEG_INFINITY;
    let nan = f16::NAN;
    let other_nan = f16::from_bits(0xfc02);

    _zzz(); // #break
}

fn _zzz() {
    ()
}
