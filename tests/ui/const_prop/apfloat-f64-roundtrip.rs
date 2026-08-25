//@ run-pass
//@ compile-flags: -O -Zmir-opt-level=3 -Cno-prepopulate-passes

// Regression test for a broken MIR optimization (issue #113407).
pub fn main() {
    let f = f64::from_bits(0x19873cc2) as f32;
    assert_eq!(f.to_bits(), 0);
}
