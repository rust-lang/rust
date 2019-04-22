// ignore-tidy-filelength

use rustc_apfloat::{Category, ExpInt, IEK_INF, IEK_NAN, IEK_ZERO};
use rustc_apfloat::{Float, FloatConvert, ParseError, Round, Status};
use rustc_apfloat::ieee::{Half, Single, Double, Quad, X87DoubleExtended};
use rustc_apfloat::unpack;

trait SingleExt {
    fn from_f32(input: f32) -> Self;
    fn to_f32(self) -> f32;
}

impl SingleExt for Single {
    fn from_f32(input: f32) -> Self {
        Self::from_bits(input.to_bits() as u128)
    }

    fn to_f32(self) -> f32 {
        f32::from_bits(self.to_bits() as u32)
    }
}

trait DoubleExt {
    fn from_f64(input: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl DoubleExt for Double {
    fn from_f64(input: f64) -> Self {
        Self::from_bits(input.to_bits() as u128)
    }

    fn to_f64(self) -> f64 {
        f64::from_bits(self.to_bits() as u64)
    }
}

#[test]
fn is_signaling() {
    // We test qNaN, -qNaN, +sNaN, -sNaN with and without payloads.
    let payload = 4;
    assert!(!Single::qnan(None).is_signaling());
    assert!(!(-Single::qnan(None)).is_signaling());
    assert!(!Single::qnan(Some(payload)).is_signaling());
    assert!(!(-Single::qnan(Some(payload))).is_signaling());
    assert!(Single::snan(None).is_signaling());
    assert!((-Single::snan(None)).is_signaling());
    assert!(Single::snan(Some(payload)).is_signaling());
    assert!((-Single::snan(Some(payload))).is_signaling());
}

#[test]
fn next() {
    // 1. Test Special Cases Values.
    //
    // Test all special values for nextUp and nextDown perscribed by IEEE-754R
    // 2008. These are:
    //   1. +inf
    //   2. -inf
    //   3. largest
    //   4. -largest
    //   5. smallest
    //   6. -smallest
    //   7. qNaN
    //   8. sNaN
    //   9. +0
    //   10. -0

    let mut status;

    // nextUp(+inf) = +inf.
    let test = unpack!(status=, Quad::INFINITY.next_up());
    let expected = Quad::INFINITY;
    assert_eq!(status, Status::OK);
    assert!(test.is_infinite());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(+inf) = -nextUp(-inf) = -(-largest) = largest
    let test = unpack!(status=, Quad::INFINITY.next_down());
    let expected = Quad::largest();
    assert_eq!(status, Status::OK);
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(-inf) = -largest
    let test = unpack!(status=, (-Quad::INFINITY).next_up());
    let expected = -Quad::largest();
    assert_eq!(status, Status::OK);
    assert!(test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(-inf) = -nextUp(+inf) = -(+inf) = -inf.
    let test = unpack!(status=, (-Quad::INFINITY).next_down());
    let expected = -Quad::INFINITY;
    assert_eq!(status, Status::OK);
    assert!(test.is_infinite() && test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(largest) = +inf
    let test = unpack!(status=, Quad::largest().next_up());
    let expected = Quad::INFINITY;
    assert_eq!(status, Status::OK);
    assert!(test.is_infinite() && !test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(largest) = -nextUp(-largest)
    //                        = -(-largest + inc)
    //                        = largest - inc.
    let test = unpack!(status=, Quad::largest().next_down());
    let expected = "0x1.fffffffffffffffffffffffffffep+16383"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_infinite() && !test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(-largest) = -largest + inc.
    let test = unpack!(status=, (-Quad::largest()).next_up());
    let expected = "-0x1.fffffffffffffffffffffffffffep+16383"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextDown(-largest) = -nextUp(largest) = -(inf) = -inf.
    let test = unpack!(status=, (-Quad::largest()).next_down());
    let expected = -Quad::INFINITY;
    assert_eq!(status, Status::OK);
    assert!(test.is_infinite() && test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(smallest) = smallest + inc.
    let test = unpack!(status=, "0x0.0000000000000000000000000001p-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "0x0.0000000000000000000000000002p-16382"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextDown(smallest) = -nextUp(-smallest) = -(-0) = +0.
    let test = unpack!(status=, "0x0.0000000000000000000000000001p-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = Quad::ZERO;
    assert_eq!(status, Status::OK);
    assert!(test.is_pos_zero());
    assert!(test.bitwise_eq(expected));

    // nextUp(-smallest) = -0.
    let test = unpack!(status=, "-0x0.0000000000000000000000000001p-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = -Quad::ZERO;
    assert_eq!(status, Status::OK);
    assert!(test.is_neg_zero());
    assert!(test.bitwise_eq(expected));

    // nextDown(-smallest) = -nextUp(smallest) = -smallest - inc.
    let test = unpack!(status=, "-0x0.0000000000000000000000000001p-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "-0x0.0000000000000000000000000002p-16382"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextUp(qNaN) = qNaN
    let test = unpack!(status=, Quad::qnan(None).next_up());
    let expected = Quad::qnan(None);
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextDown(qNaN) = qNaN
    let test = unpack!(status=, Quad::qnan(None).next_down());
    let expected = Quad::qnan(None);
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextUp(sNaN) = qNaN
    let test = unpack!(status=, Quad::snan(None).next_up());
    let expected = Quad::qnan(None);
    assert_eq!(status, Status::INVALID_OP);
    assert!(test.bitwise_eq(expected));

    // nextDown(sNaN) = qNaN
    let test = unpack!(status=, Quad::snan(None).next_down());
    let expected = Quad::qnan(None);
    assert_eq!(status, Status::INVALID_OP);
    assert!(test.bitwise_eq(expected));

    // nextUp(+0) = +smallest
    let test = unpack!(status=, Quad::ZERO.next_up());
    let expected = Quad::SMALLEST;
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextDown(+0) = -nextUp(-0) = -smallest
    let test = unpack!(status=, Quad::ZERO.next_down());
    let expected = -Quad::SMALLEST;
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextUp(-0) = +smallest
    let test = unpack!(status=, (-Quad::ZERO).next_up());
    let expected = Quad::SMALLEST;
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextDown(-0) = -nextUp(0) = -smallest
    let test = unpack!(status=, (-Quad::ZERO).next_down());
    let expected = -Quad::SMALLEST;
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // 2. Binade Boundary Tests.

    // 2a. Test denormal <-> normal binade boundaries.
    //     * nextUp(+Largest Denormal) -> +Smallest Normal.
    //     * nextDown(-Largest Denormal) -> -Smallest Normal.
    //     * nextUp(-Smallest Normal) -> -Largest Denormal.
    //     * nextDown(+Smallest Normal) -> +Largest Denormal.

    // nextUp(+Largest Denormal) -> +Smallest Normal.
    let test = unpack!(status=, "0x0.ffffffffffffffffffffffffffffp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "0x1.0000000000000000000000000000p-16382"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(test.bitwise_eq(expected));

    // nextDown(-Largest Denormal) -> -Smallest Normal.
    let test = unpack!(status=, "-0x0.ffffffffffffffffffffffffffffp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "-0x1.0000000000000000000000000000p-16382"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(test.bitwise_eq(expected));

    // nextUp(-Smallest Normal) -> -Largest Denormal.
    let test = unpack!(status=, "-0x1.0000000000000000000000000000p-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "-0x0.ffffffffffffffffffffffffffffp-16382"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(test.bitwise_eq(expected));

    // nextDown(+Smallest Normal) -> +Largest Denormal.
    let test = unpack!(status=, "+0x1.0000000000000000000000000000p-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "+0x0.ffffffffffffffffffffffffffffp-16382"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(test.bitwise_eq(expected));

    // 2b. Test normal <-> normal binade boundaries.
    //     * nextUp(-Normal Binade Boundary) -> -Normal Binade Boundary + 1.
    //     * nextDown(+Normal Binade Boundary) -> +Normal Binade Boundary - 1.
    //     * nextUp(+Normal Binade Boundary - 1) -> +Normal Binade Boundary.
    //     * nextDown(-Normal Binade Boundary + 1) -> -Normal Binade Boundary.

    // nextUp(-Normal Binade Boundary) -> -Normal Binade Boundary + 1.
    let test = unpack!(status=, "-0x1p+1".parse::<Quad>().unwrap().next_up());
    let expected = "-0x1.ffffffffffffffffffffffffffffp+0"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextDown(+Normal Binade Boundary) -> +Normal Binade Boundary - 1.
    let test = unpack!(status=, "0x1p+1".parse::<Quad>().unwrap().next_down());
    let expected = "0x1.ffffffffffffffffffffffffffffp+0"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextUp(+Normal Binade Boundary - 1) -> +Normal Binade Boundary.
    let test = unpack!(status=, "0x1.ffffffffffffffffffffffffffffp+0"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "0x1p+1".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextDown(-Normal Binade Boundary + 1) -> -Normal Binade Boundary.
    let test = unpack!(status=, "-0x1.ffffffffffffffffffffffffffffp+0"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "-0x1p+1".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // 2c. Test using next at binade boundaries with a direction away from the
    // binade boundary. Away from denormal <-> normal boundaries.
    //
    // This is to make sure that even though we are at a binade boundary, since
    // we are rounding away, we do not trigger the binade boundary code. Thus we
    // test:
    //   * nextUp(-Largest Denormal) -> -Largest Denormal + inc.
    //   * nextDown(+Largest Denormal) -> +Largest Denormal - inc.
    //   * nextUp(+Smallest Normal) -> +Smallest Normal + inc.
    //   * nextDown(-Smallest Normal) -> -Smallest Normal - inc.

    // nextUp(-Largest Denormal) -> -Largest Denormal + inc.
    let test = unpack!(status=, "-0x0.ffffffffffffffffffffffffffffp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "-0x0.fffffffffffffffffffffffffffep-16382"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(+Largest Denormal) -> +Largest Denormal - inc.
    let test = unpack!(status=, "0x0.ffffffffffffffffffffffffffffp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "0x0.fffffffffffffffffffffffffffep-16382"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(+Smallest Normal) -> +Smallest Normal + inc.
    let test = unpack!(status=, "0x1.0000000000000000000000000000p-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "0x1.0000000000000000000000000001p-16382"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(-Smallest Normal) -> -Smallest Normal - inc.
    let test = unpack!(status=, "-0x1.0000000000000000000000000000p-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "-0x1.0000000000000000000000000001p-16382"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(test.is_negative());
    assert!(test.bitwise_eq(expected));

    // 2d. Test values which cause our exponent to go to min exponent. This
    // is to ensure that guards in the code to check for min exponent
    // trigger properly.
    //     * nextUp(-0x1p-16381) -> -0x1.ffffffffffffffffffffffffffffp-16382
    //     * nextDown(-0x1.ffffffffffffffffffffffffffffp-16382) ->
    //         -0x1p-16381
    //     * nextUp(0x1.ffffffffffffffffffffffffffffp-16382) -> 0x1p-16382
    //     * nextDown(0x1p-16382) -> 0x1.ffffffffffffffffffffffffffffp-16382

    // nextUp(-0x1p-16381) -> -0x1.ffffffffffffffffffffffffffffp-16382
    let test = unpack!(status=, "-0x1p-16381".parse::<Quad>().unwrap().next_up());
    let expected = "-0x1.ffffffffffffffffffffffffffffp-16382"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextDown(-0x1.ffffffffffffffffffffffffffffp-16382) ->
    //         -0x1p-16381
    let test = unpack!(status=, "-0x1.ffffffffffffffffffffffffffffp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "-0x1p-16381".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextUp(0x1.ffffffffffffffffffffffffffffp-16382) -> 0x1p-16381
    let test = unpack!(status=, "0x1.ffffffffffffffffffffffffffffp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "0x1p-16381".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextDown(0x1p-16381) -> 0x1.ffffffffffffffffffffffffffffp-16382
    let test = unpack!(status=, "0x1p-16381".parse::<Quad>().unwrap().next_down());
    let expected = "0x1.ffffffffffffffffffffffffffffp-16382"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // 3. Now we test both denormal/normal computation which will not cause us
    // to go across binade boundaries. Specifically we test:
    //   * nextUp(+Denormal) -> +Denormal.
    //   * nextDown(+Denormal) -> +Denormal.
    //   * nextUp(-Denormal) -> -Denormal.
    //   * nextDown(-Denormal) -> -Denormal.
    //   * nextUp(+Normal) -> +Normal.
    //   * nextDown(+Normal) -> +Normal.
    //   * nextUp(-Normal) -> -Normal.
    //   * nextDown(-Normal) -> -Normal.

    // nextUp(+Denormal) -> +Denormal.
    let test = unpack!(status=, "0x0.ffffffffffffffffffffffff000cp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "0x0.ffffffffffffffffffffffff000dp-16382"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(+Denormal) -> +Denormal.
    let test = unpack!(status=, "0x0.ffffffffffffffffffffffff000cp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "0x0.ffffffffffffffffffffffff000bp-16382"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(-Denormal) -> -Denormal.
    let test = unpack!(status=, "-0x0.ffffffffffffffffffffffff000cp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "-0x0.ffffffffffffffffffffffff000bp-16382"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(-Denormal) -> -Denormal
    let test = unpack!(status=, "-0x0.ffffffffffffffffffffffff000cp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "-0x0.ffffffffffffffffffffffff000dp-16382"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(+Normal) -> +Normal.
    let test = unpack!(status=, "0x1.ffffffffffffffffffffffff000cp-16000"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "0x1.ffffffffffffffffffffffff000dp-16000"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(+Normal) -> +Normal.
    let test = unpack!(status=, "0x1.ffffffffffffffffffffffff000cp-16000"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "0x1.ffffffffffffffffffffffff000bp-16000"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(-Normal) -> -Normal.
    let test = unpack!(status=, "-0x1.ffffffffffffffffffffffff000cp-16000"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "-0x1.ffffffffffffffffffffffff000bp-16000"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(-Normal) -> -Normal.
    let test = unpack!(status=, "-0x1.ffffffffffffffffffffffff000cp-16000"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "-0x1.ffffffffffffffffffffffff000dp-16000"
        .parse::<Quad>()
        .unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(test.is_negative());
    assert!(test.bitwise_eq(expected));
}

#[test]
fn fma() {
    {
        let mut f1 = Single::from_f32(14.5);
        let f2 = Single::from_f32(-14.5);
        let f3 = Single::from_f32(225.0);
        f1 = f1.mul_add(f2, f3).value;
        assert_eq!(14.75, f1.to_f32());
    }

    {
        let val2 = Single::from_f32(2.0);
        let mut f1 = Single::from_f32(1.17549435e-38);
        let mut f2 = Single::from_f32(1.17549435e-38);
        f1 /= val2;
        f2 /= val2;
        let f3 = Single::from_f32(12.0);
        f1 = f1.mul_add(f2, f3).value;
        assert_eq!(12.0, f1.to_f32());
    }

    // Test for correct zero sign when answer is exactly zero.
    // fma(1.0, -1.0, 1.0) -> +ve 0.
    {
        let mut f1 = Double::from_f64(1.0);
        let f2 = Double::from_f64(-1.0);
        let f3 = Double::from_f64(1.0);
        f1 = f1.mul_add(f2, f3).value;
        assert!(!f1.is_negative() && f1.is_zero());
    }

    // Test for correct zero sign when answer is exactly zero and rounding towards
    // negative.
    // fma(1.0, -1.0, 1.0) -> +ve 0.
    {
        let mut f1 = Double::from_f64(1.0);
        let f2 = Double::from_f64(-1.0);
        let f3 = Double::from_f64(1.0);
        f1 = f1.mul_add_r(f2, f3, Round::TowardNegative).value;
        assert!(f1.is_negative() && f1.is_zero());
    }

    // Test for correct (in this case -ve) sign when adding like signed zeros.
    // Test fma(0.0, -0.0, -0.0) -> -ve 0.
    {
        let mut f1 = Double::from_f64(0.0);
        let f2 = Double::from_f64(-0.0);
        let f3 = Double::from_f64(-0.0);
        f1 = f1.mul_add(f2, f3).value;
        assert!(f1.is_negative() && f1.is_zero());
    }

    // Test -ve sign preservation when small negative results underflow.
    {
        let mut f1 = "-0x1p-1074".parse::<Double>().unwrap();
        let f2 = "+0x1p-1074".parse::<Double>().unwrap();
        let f3 = Double::from_f64(0.0);
        f1 = f1.mul_add(f2, f3).value;
        assert!(f1.is_negative() && f1.is_zero());
    }

    // Test x87 extended precision case from http://llvm.org/PR20728.
    {
        let mut m1 = X87DoubleExtended::from_u128(1).value;
        let m2 = X87DoubleExtended::from_u128(1).value;
        let a = X87DoubleExtended::from_u128(3).value;

        let mut loses_info = false;
        m1 = m1.mul_add(m2, a).value;
        let r: Single = m1.convert(&mut loses_info).value;
        assert!(!loses_info);
        assert_eq!(4.0, r.to_f32());
    }
}

#[test]
fn min_num() {
    let f1 = Double::from_f64(1.0);
    let f2 = Double::from_f64(2.0);
    let nan = Double::NAN;

    assert_eq!(1.0, f1.min(f2).to_f64());
    assert_eq!(1.0, f2.min(f1).to_f64());
    assert_eq!(1.0, f1.min(nan).to_f64());
    assert_eq!(1.0, nan.min(f1).to_f64());
}

#[test]
fn max_num() {
    let f1 = Double::from_f64(1.0);
    let f2 = Double::from_f64(2.0);
    let nan = Double::NAN;

    assert_eq!(2.0, f1.max(f2).to_f64());
    assert_eq!(2.0, f2.max(f1).to_f64());
    assert_eq!(1.0, f1.max(nan).to_f64());
    assert_eq!(1.0, nan.max(f1).to_f64());
}

#[test]
fn denormal() {
    // Test single precision
    {
        assert!(!Single::from_f32(0.0).is_denormal());

        let mut t = "1.17549435082228750797e-38".parse::<Single>().unwrap();
        assert!(!t.is_denormal());

        let val2 = Single::from_f32(2.0e0);
        t /= val2;
        assert!(t.is_denormal());
    }

    // Test double precision
    {
        assert!(!Double::from_f64(0.0).is_denormal());

        let mut t = "2.22507385850720138309e-308".parse::<Double>().unwrap();
        assert!(!t.is_denormal());

        let val2 = Double::from_f64(2.0e0);
        t /= val2;
        assert!(t.is_denormal());
    }

    // Test Intel double-ext
    {
        assert!(!X87DoubleExtended::from_u128(0).value.is_denormal());

        let mut t = "3.36210314311209350626e-4932"
            .parse::<X87DoubleExtended>()
            .unwrap();
        assert!(!t.is_denormal());

        t /= X87DoubleExtended::from_u128(2).value;
        assert!(t.is_denormal());
    }

    // Test quadruple precision
    {
        assert!(!Quad::from_u128(0).value.is_denormal());

        let mut t = "3.36210314311209350626267781732175260e-4932"
            .parse::<Quad>()
            .unwrap();
        assert!(!t.is_denormal());

        t /= Quad::from_u128(2).value;
        assert!(t.is_denormal());
    }
}

#[test]
fn decimal_strings_without_null_terminators() {
    // Make sure that we can parse strings without null terminators.
    // rdar://14323230.
    let val = "0.00"[..3].parse::<Double>().unwrap();
    assert_eq!(val.to_f64(), 0.0);
    let val = "0.01"[..3].parse::<Double>().unwrap();
    assert_eq!(val.to_f64(), 0.0);
    let val = "0.09"[..3].parse::<Double>().unwrap();
    assert_eq!(val.to_f64(), 0.0);
    let val = "0.095"[..4].parse::<Double>().unwrap();
    assert_eq!(val.to_f64(), 0.09);
    let val = "0.00e+3"[..7].parse::<Double>().unwrap();
    assert_eq!(val.to_f64(), 0.00);
    let val = "0e+3"[..4].parse::<Double>().unwrap();
    assert_eq!(val.to_f64(), 0.00);

}

#[test]
fn from_zero_decimal_string() {
    assert_eq!(0.0, "0".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, ".0".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+.0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-.0".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.0".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.0".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "00000.".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+00000.".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-00000.".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, ".00000".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+.00000".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-.00000".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0000.00000".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0000.00000".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0000.00000".parse::<Double>().unwrap().to_f64());
}

#[test]
fn from_zero_decimal_single_exponent_string() {
    assert_eq!(0.0, "0e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0e1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0e+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0e-1".parse::<Double>().unwrap().to_f64());


    assert_eq!(0.0, "0.e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.e1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.e+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.e-1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, ".0e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+.0e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-.0e1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, ".0e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+.0e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-.0e+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, ".0e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+.0e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-.0e-1".parse::<Double>().unwrap().to_f64());


    assert_eq!(0.0, "0.0e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.0e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.0e1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.0e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.0e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.0e+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0.0e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0.0e-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0.0e-1".parse::<Double>().unwrap().to_f64());


    assert_eq!(0.0, "000.0000e1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+000.0000e+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-000.0000e+1".parse::<Double>().unwrap().to_f64());
}

#[test]
fn from_zero_decimal_large_exponent_string() {
    assert_eq!(0.0, "0e1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0e1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0e1234".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0e+1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0e+1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0e+1234".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0e-1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0e-1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0e-1234".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "000.0000e1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "000.0000e-1234".parse::<Double>().unwrap().to_f64());
}

#[test]
fn from_zero_hexadecimal_string() {
    assert_eq!(0.0, "0x0p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0p-1".parse::<Double>().unwrap().to_f64());


    assert_eq!(0.0, "0x0.p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0.p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0.p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0.p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0.p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0.p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0.p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0.p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0.p-1".parse::<Double>().unwrap().to_f64());


    assert_eq!(0.0, "0x.0p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x.0p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x.0p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x.0p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x.0p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x.0p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x.0p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x.0p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x.0p-1".parse::<Double>().unwrap().to_f64());


    assert_eq!(0.0, "0x0.0p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0.0p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0.0p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0.0p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0.0p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0.0p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.0, "0x0.0p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "+0x0.0p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0.0p-1".parse::<Double>().unwrap().to_f64());


    assert_eq!(0.0, "0x00000.p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x0000.00000p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x.00000p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x0.p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x0p1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.0, "-0x0p1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x00000.p1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x0000.00000p1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x.00000p1234".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.0, "0x0.p1234".parse::<Double>().unwrap().to_f64());
}

#[test]
fn from_decimal_string() {
    assert_eq!(1.0, "1".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0, "2.".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.5, ".5".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.0, "1.0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-2.0, "-2".parse::<Double>().unwrap().to_f64());
    assert_eq!(-4.0, "-4.".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.5, "-.5".parse::<Double>().unwrap().to_f64());
    assert_eq!(-1.5, "-1.5".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.25e12, "1.25e12".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.25e+12, "1.25e+12".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.25e-12, "1.25e-12".parse::<Double>().unwrap().to_f64());
    assert_eq!(1024.0, "1024.".parse::<Double>().unwrap().to_f64());
    assert_eq!(1024.05, "1024.05000".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.05, ".05000".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0, "2.".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0e2, "2.e2".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0e+2, "2.e+2".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0e-2, "2.e-2".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.05e2, "002.05000e2".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.05e+2, "002.05000e+2".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.05e-2, "002.05000e-2".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.05e12, "002.05000e12".parse::<Double>().unwrap().to_f64());
    assert_eq!(
        2.05e+12,
        "002.05000e+12".parse::<Double>().unwrap().to_f64()
    );
    assert_eq!(
        2.05e-12,
        "002.05000e-12".parse::<Double>().unwrap().to_f64()
    );

    // These are "carefully selected" to overflow the fast log-base
    // calculations in the implementation.
    assert!("99e99999".parse::<Double>().unwrap().is_infinite());
    assert!("-99e99999".parse::<Double>().unwrap().is_infinite());
    assert!("1e-99999".parse::<Double>().unwrap().is_pos_zero());
    assert!("-1e-99999".parse::<Double>().unwrap().is_neg_zero());

    assert_eq!(2.71828, "2.71828".parse::<Double>().unwrap().to_f64());
}

#[test]
fn from_hexadecimal_string() {
    assert_eq!(1.0, "0x1p0".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.0, "+0x1p0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-1.0, "-0x1p0".parse::<Double>().unwrap().to_f64());

    assert_eq!(1.0, "0x1p+0".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.0, "+0x1p+0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-1.0, "-0x1p+0".parse::<Double>().unwrap().to_f64());

    assert_eq!(1.0, "0x1p-0".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.0, "+0x1p-0".parse::<Double>().unwrap().to_f64());
    assert_eq!(-1.0, "-0x1p-0".parse::<Double>().unwrap().to_f64());


    assert_eq!(2.0, "0x1p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0, "+0x1p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-2.0, "-0x1p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(2.0, "0x1p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(2.0, "+0x1p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-2.0, "-0x1p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.5, "0x1p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.5, "+0x1p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.5, "-0x1p-1".parse::<Double>().unwrap().to_f64());


    assert_eq!(3.0, "0x1.8p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(3.0, "+0x1.8p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-3.0, "-0x1.8p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(3.0, "0x1.8p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(3.0, "+0x1.8p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-3.0, "-0x1.8p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.75, "0x1.8p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.75, "+0x1.8p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.75, "-0x1.8p-1".parse::<Double>().unwrap().to_f64());


    assert_eq!(8192.0, "0x1000.000p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(8192.0, "+0x1000.000p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-8192.0, "-0x1000.000p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(8192.0, "0x1000.000p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(8192.0, "+0x1000.000p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(
        -8192.0,
        "-0x1000.000p+1".parse::<Double>().unwrap().to_f64()
    );

    assert_eq!(2048.0, "0x1000.000p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(2048.0, "+0x1000.000p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(
        -2048.0,
        "-0x1000.000p-1".parse::<Double>().unwrap().to_f64()
    );


    assert_eq!(8192.0, "0x1000p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(8192.0, "+0x1000p1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-8192.0, "-0x1000p1".parse::<Double>().unwrap().to_f64());

    assert_eq!(8192.0, "0x1000p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(8192.0, "+0x1000p+1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-8192.0, "-0x1000p+1".parse::<Double>().unwrap().to_f64());

    assert_eq!(2048.0, "0x1000p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(2048.0, "+0x1000p-1".parse::<Double>().unwrap().to_f64());
    assert_eq!(-2048.0, "-0x1000p-1".parse::<Double>().unwrap().to_f64());


    assert_eq!(16384.0, "0x10p10".parse::<Double>().unwrap().to_f64());
    assert_eq!(16384.0, "+0x10p10".parse::<Double>().unwrap().to_f64());
    assert_eq!(-16384.0, "-0x10p10".parse::<Double>().unwrap().to_f64());

    assert_eq!(16384.0, "0x10p+10".parse::<Double>().unwrap().to_f64());
    assert_eq!(16384.0, "+0x10p+10".parse::<Double>().unwrap().to_f64());
    assert_eq!(-16384.0, "-0x10p+10".parse::<Double>().unwrap().to_f64());

    assert_eq!(0.015625, "0x10p-10".parse::<Double>().unwrap().to_f64());
    assert_eq!(0.015625, "+0x10p-10".parse::<Double>().unwrap().to_f64());
    assert_eq!(-0.015625, "-0x10p-10".parse::<Double>().unwrap().to_f64());

    assert_eq!(1.0625, "0x1.1p0".parse::<Double>().unwrap().to_f64());
    assert_eq!(1.0, "0x1p0".parse::<Double>().unwrap().to_f64());

    assert_eq!(
        "0x1p-150".parse::<Double>().unwrap().to_f64(),
        "+0x800000000000000001.p-221"
            .parse::<Double>()
            .unwrap()
            .to_f64()
    );
    assert_eq!(
        2251799813685248.5,
        "0x80000000000004000000.010p-28"
            .parse::<Double>()
            .unwrap()
            .to_f64()
    );
}

#[test]
fn to_string() {
    let to_string = |d: f64, precision: usize, width: usize| {
        let x = Double::from_f64(d);
        if precision == 0 {
            format!("{:1$}", x, width)
        } else {
            format!("{:2$.1$}", x, precision, width)
        }
    };
    assert_eq!("10", to_string(10.0, 6, 3));
    assert_eq!("1.0E+1", to_string(10.0, 6, 0));
    assert_eq!("10100", to_string(1.01E+4, 5, 2));
    assert_eq!("1.01E+4", to_string(1.01E+4, 4, 2));
    assert_eq!("1.01E+4", to_string(1.01E+4, 5, 1));
    assert_eq!("0.0101", to_string(1.01E-2, 5, 2));
    assert_eq!("0.0101", to_string(1.01E-2, 4, 2));
    assert_eq!("1.01E-2", to_string(1.01E-2, 5, 1));
    assert_eq!(
        "0.78539816339744828",
        to_string(0.78539816339744830961, 0, 3)
    );
    assert_eq!(
        "4.9406564584124654E-324",
        to_string(4.9406564584124654e-324, 0, 3)
    );
    assert_eq!("873.18340000000001", to_string(873.1834, 0, 1));
    assert_eq!("8.7318340000000001E+2", to_string(873.1834, 0, 0));
    assert_eq!(
        "1.7976931348623157E+308",
        to_string(1.7976931348623157E+308, 0, 0)
    );

    let to_string = |d: f64, precision: usize, width: usize| {
        let x = Double::from_f64(d);
        if precision == 0 {
            format!("{:#1$}", x, width)
        } else {
            format!("{:#2$.1$}", x, precision, width)
        }
    };
    assert_eq!("10", to_string(10.0, 6, 3));
    assert_eq!("1.000000e+01", to_string(10.0, 6, 0));
    assert_eq!("10100", to_string(1.01E+4, 5, 2));
    assert_eq!("1.0100e+04", to_string(1.01E+4, 4, 2));
    assert_eq!("1.01000e+04", to_string(1.01E+4, 5, 1));
    assert_eq!("0.0101", to_string(1.01E-2, 5, 2));
    assert_eq!("0.0101", to_string(1.01E-2, 4, 2));
    assert_eq!("1.01000e-02", to_string(1.01E-2, 5, 1));
    assert_eq!(
        "0.78539816339744828",
        to_string(0.78539816339744830961, 0, 3)
    );
    assert_eq!(
        "4.94065645841246540e-324",
        to_string(4.9406564584124654e-324, 0, 3)
    );
    assert_eq!("873.18340000000001", to_string(873.1834, 0, 1));
    assert_eq!("8.73183400000000010e+02", to_string(873.1834, 0, 0));
    assert_eq!(
        "1.79769313486231570e+308",
        to_string(1.7976931348623157E+308, 0, 0)
    );
}

#[test]
fn to_integer() {
    let mut is_exact = false;

    assert_eq!(
        Status::OK.and(10),
        "10".parse::<Double>().unwrap().to_u128_r(
            5,
            Round::TowardZero,
            &mut is_exact,
        )
    );
    assert!(is_exact);

    assert_eq!(
        Status::INVALID_OP.and(0),
        "-10".parse::<Double>().unwrap().to_u128_r(
            5,
            Round::TowardZero,
            &mut is_exact,
        )
    );
    assert!(!is_exact);

    assert_eq!(
        Status::INVALID_OP.and(31),
        "32".parse::<Double>().unwrap().to_u128_r(
            5,
            Round::TowardZero,
            &mut is_exact,
        )
    );
    assert!(!is_exact);

    assert_eq!(
        Status::INEXACT.and(7),
        "7.9".parse::<Double>().unwrap().to_u128_r(
            5,
            Round::TowardZero,
            &mut is_exact,
        )
    );
    assert!(!is_exact);

    assert_eq!(
        Status::OK.and(-10),
        "-10".parse::<Double>().unwrap().to_i128_r(
            5,
            Round::TowardZero,
            &mut is_exact,
        )
    );
    assert!(is_exact);

    assert_eq!(
        Status::INVALID_OP.and(-16),
        "-17".parse::<Double>().unwrap().to_i128_r(
            5,
            Round::TowardZero,
            &mut is_exact,
        )
    );
    assert!(!is_exact);

    assert_eq!(
        Status::INVALID_OP.and(15),
        "16".parse::<Double>().unwrap().to_i128_r(
            5,
            Round::TowardZero,
            &mut is_exact,
        )
    );
    assert!(!is_exact);
}

#[test]
fn nan() {
    fn nanbits<T: Float>(signaling: bool, negative: bool, fill: u128) -> u128 {
        let x = if signaling {
            T::snan(Some(fill))
        } else {
            T::qnan(Some(fill))
        };
        if negative {
            (-x).to_bits()
        } else {
            x.to_bits()
        }
    }

    assert_eq!(0x7fc00000, nanbits::<Single>(false, false, 0));
    assert_eq!(0xffc00000, nanbits::<Single>(false, true, 0));
    assert_eq!(0x7fc0ae72, nanbits::<Single>(false, false, 0xae72));
    assert_eq!(0x7fffae72, nanbits::<Single>(false, false, 0xffffae72));
    assert_eq!(0x7fa00000, nanbits::<Single>(true, false, 0));
    assert_eq!(0xffa00000, nanbits::<Single>(true, true, 0));
    assert_eq!(0x7f80ae72, nanbits::<Single>(true, false, 0xae72));
    assert_eq!(0x7fbfae72, nanbits::<Single>(true, false, 0xffffae72));

    assert_eq!(0x7ff8000000000000, nanbits::<Double>(false, false, 0));
    assert_eq!(0xfff8000000000000, nanbits::<Double>(false, true, 0));
    assert_eq!(0x7ff800000000ae72, nanbits::<Double>(false, false, 0xae72));
    assert_eq!(
        0x7fffffffffffae72,
        nanbits::<Double>(false, false, 0xffffffffffffae72)
    );
    assert_eq!(0x7ff4000000000000, nanbits::<Double>(true, false, 0));
    assert_eq!(0xfff4000000000000, nanbits::<Double>(true, true, 0));
    assert_eq!(0x7ff000000000ae72, nanbits::<Double>(true, false, 0xae72));
    assert_eq!(
        0x7ff7ffffffffae72,
        nanbits::<Double>(true, false, 0xffffffffffffae72)
    );
}

#[test]
fn string_decimal_death() {
    assert_eq!(
        "".parse::<Double>(),
        Err(ParseError("Invalid string length"))
    );
    assert_eq!(
        "+".parse::<Double>(),
        Err(ParseError("String has no digits"))
    );
    assert_eq!(
        "-".parse::<Double>(),
        Err(ParseError("String has no digits"))
    );

    assert_eq!(
        "\0".parse::<Double>(),
        Err(ParseError("Invalid character in significand"))
    );
    assert_eq!(
        "1\0".parse::<Double>(),
        Err(ParseError("Invalid character in significand"))
    );
    assert_eq!(
        "1\02".parse::<Double>(),
        Err(ParseError("Invalid character in significand"))
    );
    assert_eq!(
        "1\02e1".parse::<Double>(),
        Err(ParseError("Invalid character in significand"))
    );
    assert_eq!(
        "1e\0".parse::<Double>(),
        Err(ParseError("Invalid character in exponent"))
    );
    assert_eq!(
        "1e1\0".parse::<Double>(),
        Err(ParseError("Invalid character in exponent"))
    );
    assert_eq!(
        "1e1\02".parse::<Double>(),
        Err(ParseError("Invalid character in exponent"))
    );

    assert_eq!(
        "1.0f".parse::<Double>(),
        Err(ParseError("Invalid character in significand"))
    );

    assert_eq!(
        "..".parse::<Double>(),
        Err(ParseError("String contains multiple dots"))
    );
    assert_eq!(
        "..0".parse::<Double>(),
        Err(ParseError("String contains multiple dots"))
    );
    assert_eq!(
        "1.0.0".parse::<Double>(),
        Err(ParseError("String contains multiple dots"))
    );
}

#[test]
fn string_decimal_significand_death() {
    assert_eq!(
        ".".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "+.".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "-.".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );


    assert_eq!(
        "e".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "+e".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "-e".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );

    assert_eq!(
        "e1".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "+e1".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "-e1".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );

    assert_eq!(
        ".e1".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "+.e1".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "-.e1".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );


    assert_eq!(
        ".e".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "+.e".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "-.e".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
}

#[test]
fn string_decimal_exponent_death() {
    assert_eq!(
        "1e".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "+1e".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "-1e".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );

    assert_eq!(
        "1.e".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "+1.e".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "-1.e".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );

    assert_eq!(
        ".1e".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "+.1e".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "-.1e".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );

    assert_eq!(
        "1.1e".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "+1.1e".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "-1.1e".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );


    assert_eq!(
        "1e+".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "1e-".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );

    assert_eq!(
        ".1e".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        ".1e+".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        ".1e-".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );

    assert_eq!(
        "1.0e".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "1.0e+".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "1.0e-".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
}

#[test]
fn string_hexadecimal_death() {
    assert_eq!("0x".parse::<Double>(), Err(ParseError("Invalid string")));
    assert_eq!("+0x".parse::<Double>(), Err(ParseError("Invalid string")));
    assert_eq!("-0x".parse::<Double>(), Err(ParseError("Invalid string")));

    assert_eq!(
        "0x0".parse::<Double>(),
        Err(ParseError("Hex strings require an exponent"))
    );
    assert_eq!(
        "+0x0".parse::<Double>(),
        Err(ParseError("Hex strings require an exponent"))
    );
    assert_eq!(
        "-0x0".parse::<Double>(),
        Err(ParseError("Hex strings require an exponent"))
    );

    assert_eq!(
        "0x0.".parse::<Double>(),
        Err(ParseError("Hex strings require an exponent"))
    );
    assert_eq!(
        "+0x0.".parse::<Double>(),
        Err(ParseError("Hex strings require an exponent"))
    );
    assert_eq!(
        "-0x0.".parse::<Double>(),
        Err(ParseError("Hex strings require an exponent"))
    );

    assert_eq!(
        "0x.0".parse::<Double>(),
        Err(ParseError("Hex strings require an exponent"))
    );
    assert_eq!(
        "+0x.0".parse::<Double>(),
        Err(ParseError("Hex strings require an exponent"))
    );
    assert_eq!(
        "-0x.0".parse::<Double>(),
        Err(ParseError("Hex strings require an exponent"))
    );

    assert_eq!(
        "0x0.0".parse::<Double>(),
        Err(ParseError("Hex strings require an exponent"))
    );
    assert_eq!(
        "+0x0.0".parse::<Double>(),
        Err(ParseError("Hex strings require an exponent"))
    );
    assert_eq!(
        "-0x0.0".parse::<Double>(),
        Err(ParseError("Hex strings require an exponent"))
    );

    assert_eq!(
        "0x\0".parse::<Double>(),
        Err(ParseError("Invalid character in significand"))
    );
    assert_eq!(
        "0x1\0".parse::<Double>(),
        Err(ParseError("Invalid character in significand"))
    );
    assert_eq!(
        "0x1\02".parse::<Double>(),
        Err(ParseError("Invalid character in significand"))
    );
    assert_eq!(
        "0x1\02p1".parse::<Double>(),
        Err(ParseError("Invalid character in significand"))
    );
    assert_eq!(
        "0x1p\0".parse::<Double>(),
        Err(ParseError("Invalid character in exponent"))
    );
    assert_eq!(
        "0x1p1\0".parse::<Double>(),
        Err(ParseError("Invalid character in exponent"))
    );
    assert_eq!(
        "0x1p1\02".parse::<Double>(),
        Err(ParseError("Invalid character in exponent"))
    );

    assert_eq!(
        "0x1p0f".parse::<Double>(),
        Err(ParseError("Invalid character in exponent"))
    );

    assert_eq!(
        "0x..p1".parse::<Double>(),
        Err(ParseError("String contains multiple dots"))
    );
    assert_eq!(
        "0x..0p1".parse::<Double>(),
        Err(ParseError("String contains multiple dots"))
    );
    assert_eq!(
        "0x1.0.0p1".parse::<Double>(),
        Err(ParseError("String contains multiple dots"))
    );
}

#[test]
fn string_hexadecimal_significand_death() {
    assert_eq!(
        "0x.".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "+0x.".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "-0x.".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );

    assert_eq!(
        "0xp".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "+0xp".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "-0xp".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );

    assert_eq!(
        "0xp+".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "+0xp+".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "-0xp+".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );

    assert_eq!(
        "0xp-".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "+0xp-".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "-0xp-".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );


    assert_eq!(
        "0x.p".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "+0x.p".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "-0x.p".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );

    assert_eq!(
        "0x.p+".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "+0x.p+".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "-0x.p+".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );

    assert_eq!(
        "0x.p-".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "+0x.p-".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
    assert_eq!(
        "-0x.p-".parse::<Double>(),
        Err(ParseError("Significand has no digits"))
    );
}

#[test]
fn string_hexadecimal_exponent_death() {
    assert_eq!(
        "0x1p".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "+0x1p".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "-0x1p".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );

    assert_eq!(
        "0x1p+".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "+0x1p+".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "-0x1p+".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );

    assert_eq!(
        "0x1p-".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "+0x1p-".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "-0x1p-".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );


    assert_eq!(
        "0x1.p".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "+0x1.p".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "-0x1.p".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );

    assert_eq!(
        "0x1.p+".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "+0x1.p+".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "-0x1.p+".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );

    assert_eq!(
        "0x1.p-".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "+0x1.p-".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "-0x1.p-".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );


    assert_eq!(
        "0x.1p".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "+0x.1p".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "-0x.1p".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );

    assert_eq!(
        "0x.1p+".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "+0x.1p+".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "-0x.1p+".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );

    assert_eq!(
        "0x.1p-".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "+0x.1p-".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "-0x.1p-".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );


    assert_eq!(
        "0x1.1p".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "+0x1.1p".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "-0x1.1p".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );

    assert_eq!(
        "0x1.1p+".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "+0x1.1p+".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "-0x1.1p+".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );

    assert_eq!(
        "0x1.1p-".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "+0x1.1p-".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
    assert_eq!(
        "-0x1.1p-".parse::<Double>(),
        Err(ParseError("Exponent has no digits"))
    );
}

#[test]
fn exact_inverse() {
    // Trivial operation.
    assert!(
        Double::from_f64(2.0)
            .get_exact_inverse()
            .unwrap()
            .bitwise_eq(Double::from_f64(0.5))
    );
    assert!(
        Single::from_f32(2.0)
            .get_exact_inverse()
            .unwrap()
            .bitwise_eq(Single::from_f32(0.5))
    );
    assert!(
        "2.0"
            .parse::<Quad>()
            .unwrap()
            .get_exact_inverse()
            .unwrap()
            .bitwise_eq("0.5".parse::<Quad>().unwrap())
    );
    assert!(
        "2.0"
            .parse::<X87DoubleExtended>()
            .unwrap()
            .get_exact_inverse()
            .unwrap()
            .bitwise_eq("0.5".parse::<X87DoubleExtended>().unwrap())
    );

    // FLT_MIN
    assert!(
        Single::from_f32(1.17549435e-38)
            .get_exact_inverse()
            .unwrap()
            .bitwise_eq(Single::from_f32(8.5070592e+37))
    );

    // Large float, inverse is a denormal.
    assert!(Single::from_f32(1.7014118e38).get_exact_inverse().is_none());
    // Zero
    assert!(Double::from_f64(0.0).get_exact_inverse().is_none());
    // Denormalized float
    assert!(
        Single::from_f32(1.40129846e-45)
            .get_exact_inverse()
            .is_none()
    );
}

#[test]
fn round_to_integral() {
    let t = Double::from_f64(-0.5);
    assert_eq!(-0.0, t.round_to_integral(Round::TowardZero).value.to_f64());
    assert_eq!(
        -1.0,
        t.round_to_integral(Round::TowardNegative).value.to_f64()
    );
    assert_eq!(
        -0.0,
        t.round_to_integral(Round::TowardPositive).value.to_f64()
    );
    assert_eq!(
        -0.0,
        t.round_to_integral(Round::NearestTiesToEven).value.to_f64()
    );

    let s = Double::from_f64(3.14);
    assert_eq!(3.0, s.round_to_integral(Round::TowardZero).value.to_f64());
    assert_eq!(
        3.0,
        s.round_to_integral(Round::TowardNegative).value.to_f64()
    );
    assert_eq!(
        4.0,
        s.round_to_integral(Round::TowardPositive).value.to_f64()
    );
    assert_eq!(
        3.0,
        s.round_to_integral(Round::NearestTiesToEven).value.to_f64()
    );

    let r = Double::largest();
    assert_eq!(
        r.to_f64(),
        r.round_to_integral(Round::TowardZero).value.to_f64()
    );
    assert_eq!(
        r.to_f64(),
        r.round_to_integral(Round::TowardNegative).value.to_f64()
    );
    assert_eq!(
        r.to_f64(),
        r.round_to_integral(Round::TowardPositive).value.to_f64()
    );
    assert_eq!(
        r.to_f64(),
        r.round_to_integral(Round::NearestTiesToEven).value.to_f64()
    );

    let p = Double::ZERO.round_to_integral(Round::TowardZero).value;
    assert_eq!(0.0, p.to_f64());
    let p = (-Double::ZERO).round_to_integral(Round::TowardZero).value;
    assert_eq!(-0.0, p.to_f64());
    let p = Double::NAN.round_to_integral(Round::TowardZero).value;
    assert!(p.to_f64().is_nan());
    let p = Double::INFINITY.round_to_integral(Round::TowardZero).value;
    assert!(p.to_f64().is_infinite() && p.to_f64() > 0.0);
    let p = (-Double::INFINITY)
        .round_to_integral(Round::TowardZero)
        .value;
    assert!(p.to_f64().is_infinite() && p.to_f64() < 0.0);
}

#[test]
fn is_integer() {
    let t = Double::from_f64(-0.0);
    assert!(t.is_integer());
    let t = Double::from_f64(3.14159);
    assert!(!t.is_integer());
    let t = Double::NAN;
    assert!(!t.is_integer());
    let t = Double::INFINITY;
    assert!(!t.is_integer());
    let t = -Double::INFINITY;
    assert!(!t.is_integer());
    let t = Double::largest();
    assert!(t.is_integer());
}

#[test]
fn largest() {
    assert_eq!(3.402823466e+38, Single::largest().to_f32());
    assert_eq!(1.7976931348623158e+308, Double::largest().to_f64());
}

#[test]
fn smallest() {
    let test = Single::SMALLEST;
    let expected = "0x0.000002p-126".parse::<Single>().unwrap();
    assert!(!test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(test.is_denormal());
    assert!(test.bitwise_eq(expected));

    let test = -Single::SMALLEST;
    let expected = "-0x0.000002p-126".parse::<Single>().unwrap();
    assert!(test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(test.is_denormal());
    assert!(test.bitwise_eq(expected));

    let test = Quad::SMALLEST;
    let expected = "0x0.0000000000000000000000000001p-16382"
        .parse::<Quad>()
        .unwrap();
    assert!(!test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(test.is_denormal());
    assert!(test.bitwise_eq(expected));

    let test = -Quad::SMALLEST;
    let expected = "-0x0.0000000000000000000000000001p-16382"
        .parse::<Quad>()
        .unwrap();
    assert!(test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(test.is_denormal());
    assert!(test.bitwise_eq(expected));
}

#[test]
fn smallest_normalized() {
    let test = Single::smallest_normalized();
    let expected = "0x1p-126".parse::<Single>().unwrap();
    assert!(!test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(!test.is_denormal());
    assert!(test.bitwise_eq(expected));

    let test = -Single::smallest_normalized();
    let expected = "-0x1p-126".parse::<Single>().unwrap();
    assert!(test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(!test.is_denormal());
    assert!(test.bitwise_eq(expected));

    let test = Quad::smallest_normalized();
    let expected = "0x1p-16382".parse::<Quad>().unwrap();
    assert!(!test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(!test.is_denormal());
    assert!(test.bitwise_eq(expected));

    let test = -Quad::smallest_normalized();
    let expected = "-0x1p-16382".parse::<Quad>().unwrap();
    assert!(test.is_negative());
    assert!(test.is_finite_non_zero());
    assert!(!test.is_denormal());
    assert!(test.bitwise_eq(expected));
}

#[test]
fn zero() {
    assert_eq!(0.0, Single::from_f32(0.0).to_f32());
    assert_eq!(-0.0, Single::from_f32(-0.0).to_f32());
    assert!(Single::from_f32(-0.0).is_negative());

    assert_eq!(0.0, Double::from_f64(0.0).to_f64());
    assert_eq!(-0.0, Double::from_f64(-0.0).to_f64());
    assert!(Double::from_f64(-0.0).is_negative());

    fn test<T: Float>(sign: bool, bits: u128) {
        let test = if sign { -T::ZERO } else { T::ZERO };
        let pattern = if sign { "-0x0p+0" } else { "0x0p+0" };
        let expected = pattern.parse::<T>().unwrap();
        assert!(test.is_zero());
        assert_eq!(sign, test.is_negative());
        assert!(test.bitwise_eq(expected));
        assert_eq!(bits, test.to_bits());
    }
    test::<Half>(false, 0);
    test::<Half>(true, 0x8000);
    test::<Single>(false, 0);
    test::<Single>(true, 0x80000000);
    test::<Double>(false, 0);
    test::<Double>(true, 0x8000000000000000);
    test::<Quad>(false, 0);
    test::<Quad>(true, 0x8000000000000000_0000000000000000);
    test::<X87DoubleExtended>(false, 0);
    test::<X87DoubleExtended>(true, 0x8000_0000000000000000);
}

#[test]
fn copy_sign() {
    assert!(Double::from_f64(-42.0).bitwise_eq(
        Double::from_f64(42.0).copy_sign(
            Double::from_f64(-1.0),
        ),
    ));
    assert!(Double::from_f64(42.0).bitwise_eq(
        Double::from_f64(-42.0).copy_sign(
            Double::from_f64(1.0),
        ),
    ));
    assert!(Double::from_f64(-42.0).bitwise_eq(
        Double::from_f64(-42.0).copy_sign(
            Double::from_f64(-1.0),
        ),
    ));
    assert!(Double::from_f64(42.0).bitwise_eq(
        Double::from_f64(42.0).copy_sign(
            Double::from_f64(1.0),
        ),
    ));
}

#[test]
fn convert() {
    let mut loses_info = false;
    let test = "1.0".parse::<Double>().unwrap();
    let test: Single = test.convert(&mut loses_info).value;
    assert_eq!(1.0, test.to_f32());
    assert!(!loses_info);

    let mut test = "0x1p-53".parse::<X87DoubleExtended>().unwrap();
    let one = "1.0".parse::<X87DoubleExtended>().unwrap();
    test += one;
    let test: Double = test.convert(&mut loses_info).value;
    assert_eq!(1.0, test.to_f64());
    assert!(loses_info);

    let mut test = "0x1p-53".parse::<Quad>().unwrap();
    let one = "1.0".parse::<Quad>().unwrap();
    test += one;
    let test: Double = test.convert(&mut loses_info).value;
    assert_eq!(1.0, test.to_f64());
    assert!(loses_info);

    let test = "0xf.fffffffp+28".parse::<X87DoubleExtended>().unwrap();
    let test: Double = test.convert(&mut loses_info).value;
    assert_eq!(4294967295.0, test.to_f64());
    assert!(!loses_info);

    let test = Single::snan(None);
    let x87_snan = X87DoubleExtended::snan(None);
    let test: X87DoubleExtended = test.convert(&mut loses_info).value;
    assert!(test.bitwise_eq(x87_snan));
    assert!(!loses_info);

    let test = Single::qnan(None);
    let x87_qnan = X87DoubleExtended::qnan(None);
    let test: X87DoubleExtended = test.convert(&mut loses_info).value;
    assert!(test.bitwise_eq(x87_qnan));
    assert!(!loses_info);

    let test = X87DoubleExtended::snan(None);
    let test: X87DoubleExtended = test.convert(&mut loses_info).value;
    assert!(test.bitwise_eq(x87_snan));
    assert!(!loses_info);

    let test = X87DoubleExtended::qnan(None);
    let test: X87DoubleExtended = test.convert(&mut loses_info).value;
    assert!(test.bitwise_eq(x87_qnan));
    assert!(!loses_info);
}

#[test]
fn is_negative() {
    let t = "0x1p+0".parse::<Single>().unwrap();
    assert!(!t.is_negative());
    let t = "-0x1p+0".parse::<Single>().unwrap();
    assert!(t.is_negative());

    assert!(!Single::INFINITY.is_negative());
    assert!((-Single::INFINITY).is_negative());

    assert!(!Single::ZERO.is_negative());
    assert!((-Single::ZERO).is_negative());

    assert!(!Single::NAN.is_negative());
    assert!((-Single::NAN).is_negative());

    assert!(!Single::snan(None).is_negative());
    assert!((-Single::snan(None)).is_negative());
}

#[test]
fn is_normal() {
    let t = "0x1p+0".parse::<Single>().unwrap();
    assert!(t.is_normal());

    assert!(!Single::INFINITY.is_normal());
    assert!(!Single::ZERO.is_normal());
    assert!(!Single::NAN.is_normal());
    assert!(!Single::snan(None).is_normal());
    assert!(!"0x1p-149".parse::<Single>().unwrap().is_normal());
}

#[test]
fn is_finite() {
    let t = "0x1p+0".parse::<Single>().unwrap();
    assert!(t.is_finite());
    assert!(!Single::INFINITY.is_finite());
    assert!(Single::ZERO.is_finite());
    assert!(!Single::NAN.is_finite());
    assert!(!Single::snan(None).is_finite());
    assert!("0x1p-149".parse::<Single>().unwrap().is_finite());
}

#[test]
fn is_infinite() {
    let t = "0x1p+0".parse::<Single>().unwrap();
    assert!(!t.is_infinite());
    assert!(Single::INFINITY.is_infinite());
    assert!(!Single::ZERO.is_infinite());
    assert!(!Single::NAN.is_infinite());
    assert!(!Single::snan(None).is_infinite());
    assert!(!"0x1p-149".parse::<Single>().unwrap().is_infinite());
}

#[test]
fn is_nan() {
    let t = "0x1p+0".parse::<Single>().unwrap();
    assert!(!t.is_nan());
    assert!(!Single::INFINITY.is_nan());
    assert!(!Single::ZERO.is_nan());
    assert!(Single::NAN.is_nan());
    assert!(Single::snan(None).is_nan());
    assert!(!"0x1p-149".parse::<Single>().unwrap().is_nan());
}

#[test]
fn is_finite_non_zero() {
    // Test positive/negative normal value.
    assert!("0x1p+0".parse::<Single>().unwrap().is_finite_non_zero());
    assert!("-0x1p+0".parse::<Single>().unwrap().is_finite_non_zero());

    // Test positive/negative denormal value.
    assert!("0x1p-149".parse::<Single>().unwrap().is_finite_non_zero());
    assert!("-0x1p-149".parse::<Single>().unwrap().is_finite_non_zero());

    // Test +/- Infinity.
    assert!(!Single::INFINITY.is_finite_non_zero());
    assert!(!(-Single::INFINITY).is_finite_non_zero());

    // Test +/- Zero.
    assert!(!Single::ZERO.is_finite_non_zero());
    assert!(!(-Single::ZERO).is_finite_non_zero());

    // Test +/- qNaN. +/- don't mean anything with qNaN but paranoia can't hurt in
    // this instance.
    assert!(!Single::NAN.is_finite_non_zero());
    assert!(!(-Single::NAN).is_finite_non_zero());

    // Test +/- sNaN. +/- don't mean anything with sNaN but paranoia can't hurt in
    // this instance.
    assert!(!Single::snan(None).is_finite_non_zero());
    assert!(!(-Single::snan(None)).is_finite_non_zero());
}

#[test]
fn add() {
    // Test Special Cases against each other and normal values.

    // FIXMES/NOTES:
    // 1. Since we perform only default exception handling all operations with
    // signaling NaNs should have a result that is a quiet NaN. Currently they
    // return sNaN.

    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let qnan = Single::NAN;
    let p_normal_value = "0x1p+0".parse::<Single>().unwrap();
    let m_normal_value = "-0x1p+0".parse::<Single>().unwrap();
    let p_largest_value = Single::largest();
    let m_largest_value = -Single::largest();
    let p_smallest_value = Single::SMALLEST;
    let m_smallest_value = -Single::SMALLEST;
    let p_smallest_normalized = Single::smallest_normalized();
    let m_smallest_normalized = -Single::smallest_normalized();

    let overflow_status = Status::OVERFLOW | Status::INEXACT;

    let special_cases = [
        (p_inf, p_inf, "inf", Status::OK, Category::Infinity),
        (p_inf, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, p_zero, "inf", Status::OK, Category::Infinity),
        (p_inf, m_zero, "inf", Status::OK, Category::Infinity),
        (p_inf, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(p_inf, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (p_inf, p_normal_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_normal_value, "inf", Status::OK, Category::Infinity),
        (
            p_inf,
            p_largest_value,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            m_largest_value,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            p_smallest_value,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            m_smallest_value,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            p_smallest_normalized,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            m_smallest_normalized,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (m_inf, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, m_inf, "-inf", Status::OK, Category::Infinity),
        (m_inf, p_zero, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_zero, "-inf", Status::OK, Category::Infinity),
        (m_inf, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(m_inf, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_inf,
            p_normal_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            m_normal_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            p_largest_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            m_largest_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            p_smallest_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            m_smallest_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            p_smallest_normalized,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            m_smallest_normalized,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (p_zero, p_inf, "inf", Status::OK, Category::Infinity),
        (p_zero, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_zero, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(p_zero, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            p_zero,
            p_normal_value,
            "0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            p_zero,
            m_normal_value,
            "-0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            p_zero,
            p_largest_value,
            "0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            p_zero,
            m_largest_value,
            "-0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            p_zero,
            p_smallest_value,
            "0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            p_zero,
            m_smallest_value,
            "-0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            p_zero,
            p_smallest_normalized,
            "0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            p_zero,
            m_smallest_normalized,
            "-0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (m_zero, p_inf, "inf", Status::OK, Category::Infinity),
        (m_zero, m_inf, "-inf", Status::OK, Category::Infinity),
        (m_zero, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_zero, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(m_zero, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_zero,
            p_normal_value,
            "0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            m_zero,
            m_normal_value,
            "-0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            m_zero,
            p_largest_value,
            "0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            m_zero,
            m_largest_value,
            "-0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            m_zero,
            p_smallest_value,
            "0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            m_zero,
            m_smallest_value,
            "-0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            m_zero,
            p_smallest_normalized,
            "0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            m_zero,
            m_smallest_normalized,
            "-0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (qnan, p_inf, "nan", Status::OK, Category::NaN),
        (qnan, m_inf, "nan", Status::OK, Category::NaN),
        (qnan, p_zero, "nan", Status::OK, Category::NaN),
        (qnan, m_zero, "nan", Status::OK, Category::NaN),
        (qnan, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(qnan, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (qnan, p_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, m_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, p_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_value, "nan", Status::OK, Category::NaN),
        (
            qnan,
            p_smallest_normalized,
            "nan",
            Status::OK,
            Category::NaN,
        ),
        (
            qnan,
            m_smallest_normalized,
            "nan",
            Status::OK,
            Category::NaN,
        ),
        /*
// See Note 1.
(snan, p_inf, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_inf, "nan", Status::INVALID_OP, Category::NaN),
(snan, p_zero, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_zero, "nan", Status::INVALID_OP, Category::NaN),
(snan, qnan, "nan", Status::INVALID_OP, Category::NaN),
(snan, snan, "nan", Status::INVALID_OP, Category::NaN),
(snan, p_normal_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_normal_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, p_largest_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_largest_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, p_smallest_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_smallest_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, p_smallest_normalized, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_smallest_normalized, "nan", Status::INVALID_OP, Category::NaN),
        */
        (p_normal_value, p_inf, "inf", Status::OK, Category::Infinity),
        (
            p_normal_value,
            m_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_normal_value,
            p_zero,
            "0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            p_normal_value,
            m_zero,
            "0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (p_normal_value, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(p_normal_value, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            p_normal_value,
            p_normal_value,
            "0x1p+1",
            Status::OK,
            Category::Normal,
        ),
        (
            p_normal_value,
            m_normal_value,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_normal_value,
            p_largest_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_normal_value,
            m_largest_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_normal_value,
            p_smallest_value,
            "0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_normal_value,
            m_smallest_value,
            "0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_normal_value,
            p_smallest_normalized,
            "0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_normal_value,
            m_smallest_normalized,
            "0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (m_normal_value, p_inf, "inf", Status::OK, Category::Infinity),
        (
            m_normal_value,
            m_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_normal_value,
            p_zero,
            "-0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            m_normal_value,
            m_zero,
            "-0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (m_normal_value, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(m_normal_value, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_normal_value,
            p_normal_value,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_normal_value,
            m_normal_value,
            "-0x1p+1",
            Status::OK,
            Category::Normal,
        ),
        (
            m_normal_value,
            p_largest_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_normal_value,
            m_largest_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_normal_value,
            p_smallest_value,
            "-0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_normal_value,
            m_smallest_value,
            "-0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_normal_value,
            p_smallest_normalized,
            "-0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_normal_value,
            m_smallest_normalized,
            "-0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_largest_value,
            p_inf,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_largest_value,
            m_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_largest_value,
            p_zero,
            "0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            p_largest_value,
            m_zero,
            "0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (p_largest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(p_largest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            p_largest_value,
            p_normal_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_largest_value,
            m_normal_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_largest_value,
            p_largest_value,
            "inf",
            overflow_status,
            Category::Infinity,
        ),
        (
            p_largest_value,
            m_largest_value,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_largest_value,
            p_smallest_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_largest_value,
            m_smallest_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_largest_value,
            p_smallest_normalized,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_largest_value,
            m_smallest_normalized,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_largest_value,
            p_inf,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_largest_value,
            m_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_largest_value,
            p_zero,
            "-0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            m_largest_value,
            m_zero,
            "-0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (m_largest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(m_largest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_largest_value,
            p_normal_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_largest_value,
            m_normal_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_largest_value,
            p_largest_value,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_largest_value,
            m_largest_value,
            "-inf",
            overflow_status,
            Category::Infinity,
        ),
        (
            m_largest_value,
            p_smallest_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_largest_value,
            m_smallest_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_largest_value,
            p_smallest_normalized,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_largest_value,
            m_smallest_normalized,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_smallest_value,
            p_inf,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_smallest_value,
            m_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_smallest_value,
            p_zero,
            "0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_value,
            m_zero,
            "0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (p_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(p_smallest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            p_smallest_value,
            p_normal_value,
            "0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_smallest_value,
            m_normal_value,
            "-0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_smallest_value,
            p_largest_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_smallest_value,
            m_largest_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_smallest_value,
            p_smallest_value,
            "0x1p-148",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_value,
            m_smallest_value,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_smallest_value,
            p_smallest_normalized,
            "0x1.000002p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_value,
            m_smallest_normalized,
            "-0x1.fffffcp-127",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_value,
            p_inf,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_smallest_value,
            m_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_smallest_value,
            p_zero,
            "-0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_value,
            m_zero,
            "-0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (m_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(m_smallest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_smallest_value,
            p_normal_value,
            "0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_smallest_value,
            m_normal_value,
            "-0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_smallest_value,
            p_largest_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_smallest_value,
            m_largest_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_smallest_value,
            p_smallest_value,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_smallest_value,
            m_smallest_value,
            "-0x1p-148",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_value,
            p_smallest_normalized,
            "0x1.fffffcp-127",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_value,
            m_smallest_normalized,
            "-0x1.000002p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            p_inf,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_smallest_normalized,
            m_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_smallest_normalized,
            p_zero,
            "0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            m_zero,
            "0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            qnan,
            "nan",
            Status::OK,
            Category::NaN,
        ),
        /*
// See Note 1.
(p_smallest_normalized, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            p_smallest_normalized,
            p_normal_value,
            "0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            m_normal_value,
            "-0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            p_largest_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            m_largest_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            p_smallest_value,
            "0x1.000002p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            m_smallest_value,
            "0x1.fffffcp-127",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            p_smallest_normalized,
            "0x1p-125",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            m_smallest_normalized,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_smallest_normalized,
            p_inf,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_smallest_normalized,
            m_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_smallest_normalized,
            p_zero,
            "-0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            m_zero,
            "-0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            qnan,
            "nan",
            Status::OK,
            Category::NaN,
        ),
        /*
// See Note 1.
(m_smallest_normalized, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_smallest_normalized,
            p_normal_value,
            "0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            m_normal_value,
            "-0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            p_largest_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            m_largest_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            p_smallest_value,
            "-0x1.fffffcp-127",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            m_smallest_value,
            "-0x1.000002p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            p_smallest_normalized,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_smallest_normalized,
            m_smallest_normalized,
            "-0x1p-125",
            Status::OK,
            Category::Normal,
        ),
    ];

    for &(x, y, e_result, e_status, e_category) in &special_cases[..] {
        let status;
        let result = unpack!(status=, x + y);
        assert_eq!(status, e_status);
        assert_eq!(result.category(), e_category);
        assert!(result.bitwise_eq(e_result.parse::<Single>().unwrap()));
    }
}

#[test]
fn subtract() {
    // Test Special Cases against each other and normal values.

    // FIXMES/NOTES:
    // 1. Since we perform only default exception handling all operations with
    // signaling NaNs should have a result that is a quiet NaN. Currently they
    // return sNaN.

    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let qnan = Single::NAN;
    let p_normal_value = "0x1p+0".parse::<Single>().unwrap();
    let m_normal_value = "-0x1p+0".parse::<Single>().unwrap();
    let p_largest_value = Single::largest();
    let m_largest_value = -Single::largest();
    let p_smallest_value = Single::SMALLEST;
    let m_smallest_value = -Single::SMALLEST;
    let p_smallest_normalized = Single::smallest_normalized();
    let m_smallest_normalized = -Single::smallest_normalized();

    let overflow_status = Status::OVERFLOW | Status::INEXACT;

    let special_cases = [
        (p_inf, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, m_inf, "inf", Status::OK, Category::Infinity),
        (p_inf, p_zero, "inf", Status::OK, Category::Infinity),
        (p_inf, m_zero, "inf", Status::OK, Category::Infinity),
        (p_inf, qnan, "-nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(p_inf, snan, "-nan", Status::INVALID_OP, Category::NaN),
        */
        (p_inf, p_normal_value, "inf", Status::OK, Category::Infinity),
        (p_inf, m_normal_value, "inf", Status::OK, Category::Infinity),
        (
            p_inf,
            p_largest_value,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            m_largest_value,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            p_smallest_value,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            m_smallest_value,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            p_smallest_normalized,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            m_smallest_normalized,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (m_inf, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, p_zero, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_zero, "-inf", Status::OK, Category::Infinity),
        (m_inf, qnan, "-nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(m_inf, snan, "-nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_inf,
            p_normal_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            m_normal_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            p_largest_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            m_largest_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            p_smallest_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            m_smallest_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            p_smallest_normalized,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            m_smallest_normalized,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (p_zero, p_inf, "-inf", Status::OK, Category::Infinity),
        (p_zero, m_inf, "inf", Status::OK, Category::Infinity),
        (p_zero, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, qnan, "-nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(p_zero, snan, "-nan", Status::INVALID_OP, Category::NaN),
        */
        (
            p_zero,
            p_normal_value,
            "-0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            p_zero,
            m_normal_value,
            "0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            p_zero,
            p_largest_value,
            "-0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            p_zero,
            m_largest_value,
            "0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            p_zero,
            p_smallest_value,
            "-0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            p_zero,
            m_smallest_value,
            "0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            p_zero,
            p_smallest_normalized,
            "-0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            p_zero,
            m_smallest_normalized,
            "0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (m_zero, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_zero, m_inf, "inf", Status::OK, Category::Infinity),
        (m_zero, p_zero, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, qnan, "-nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(m_zero, snan, "-nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_zero,
            p_normal_value,
            "-0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            m_zero,
            m_normal_value,
            "0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            m_zero,
            p_largest_value,
            "-0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            m_zero,
            m_largest_value,
            "0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            m_zero,
            p_smallest_value,
            "-0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            m_zero,
            m_smallest_value,
            "0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            m_zero,
            p_smallest_normalized,
            "-0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            m_zero,
            m_smallest_normalized,
            "0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (qnan, p_inf, "nan", Status::OK, Category::NaN),
        (qnan, m_inf, "nan", Status::OK, Category::NaN),
        (qnan, p_zero, "nan", Status::OK, Category::NaN),
        (qnan, m_zero, "nan", Status::OK, Category::NaN),
        (qnan, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(qnan, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (qnan, p_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, m_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, p_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_value, "nan", Status::OK, Category::NaN),
        (
            qnan,
            p_smallest_normalized,
            "nan",
            Status::OK,
            Category::NaN,
        ),
        (
            qnan,
            m_smallest_normalized,
            "nan",
            Status::OK,
            Category::NaN,
        ),
        /*
// See Note 1.
(snan, p_inf, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_inf, "nan", Status::INVALID_OP, Category::NaN),
(snan, p_zero, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_zero, "nan", Status::INVALID_OP, Category::NaN),
(snan, qnan, "nan", Status::INVALID_OP, Category::NaN),
(snan, snan, "nan", Status::INVALID_OP, Category::NaN),
(snan, p_normal_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_normal_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, p_largest_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_largest_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, p_smallest_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_smallest_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, p_smallest_normalized, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_smallest_normalized, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            p_normal_value,
            p_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (p_normal_value, m_inf, "inf", Status::OK, Category::Infinity),
        (
            p_normal_value,
            p_zero,
            "0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            p_normal_value,
            m_zero,
            "0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (p_normal_value, qnan, "-nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(p_normal_value, snan, "-nan", Status::INVALID_OP, Category::NaN),
        */
        (
            p_normal_value,
            p_normal_value,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_normal_value,
            m_normal_value,
            "0x1p+1",
            Status::OK,
            Category::Normal,
        ),
        (
            p_normal_value,
            p_largest_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_normal_value,
            m_largest_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_normal_value,
            p_smallest_value,
            "0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_normal_value,
            m_smallest_value,
            "0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_normal_value,
            p_smallest_normalized,
            "0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_normal_value,
            m_smallest_normalized,
            "0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_normal_value,
            p_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (m_normal_value, m_inf, "inf", Status::OK, Category::Infinity),
        (
            m_normal_value,
            p_zero,
            "-0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            m_normal_value,
            m_zero,
            "-0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (m_normal_value, qnan, "-nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(m_normal_value, snan, "-nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_normal_value,
            p_normal_value,
            "-0x1p+1",
            Status::OK,
            Category::Normal,
        ),
        (
            m_normal_value,
            m_normal_value,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_normal_value,
            p_largest_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_normal_value,
            m_largest_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_normal_value,
            p_smallest_value,
            "-0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_normal_value,
            m_smallest_value,
            "-0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_normal_value,
            p_smallest_normalized,
            "-0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_normal_value,
            m_smallest_normalized,
            "-0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_largest_value,
            p_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_largest_value,
            m_inf,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_largest_value,
            p_zero,
            "0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            p_largest_value,
            m_zero,
            "0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (p_largest_value, qnan, "-nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(p_largest_value, snan, "-nan", Status::INVALID_OP, Category::NaN),
        */
        (
            p_largest_value,
            p_normal_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_largest_value,
            m_normal_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_largest_value,
            p_largest_value,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_largest_value,
            m_largest_value,
            "inf",
            overflow_status,
            Category::Infinity,
        ),
        (
            p_largest_value,
            p_smallest_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_largest_value,
            m_smallest_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_largest_value,
            p_smallest_normalized,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_largest_value,
            m_smallest_normalized,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_largest_value,
            p_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_largest_value,
            m_inf,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_largest_value,
            p_zero,
            "-0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            m_largest_value,
            m_zero,
            "-0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (m_largest_value, qnan, "-nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(m_largest_value, snan, "-nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_largest_value,
            p_normal_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_largest_value,
            m_normal_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_largest_value,
            p_largest_value,
            "-inf",
            overflow_status,
            Category::Infinity,
        ),
        (
            m_largest_value,
            m_largest_value,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_largest_value,
            p_smallest_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_largest_value,
            m_smallest_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_largest_value,
            p_smallest_normalized,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_largest_value,
            m_smallest_normalized,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_smallest_value,
            p_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_smallest_value,
            m_inf,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_smallest_value,
            p_zero,
            "0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_value,
            m_zero,
            "0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (p_smallest_value, qnan, "-nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(p_smallest_value, snan, "-nan", Status::INVALID_OP, Category::NaN),
        */
        (
            p_smallest_value,
            p_normal_value,
            "-0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_smallest_value,
            m_normal_value,
            "0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_smallest_value,
            p_largest_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_smallest_value,
            m_largest_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_smallest_value,
            p_smallest_value,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_smallest_value,
            m_smallest_value,
            "0x1p-148",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_value,
            p_smallest_normalized,
            "-0x1.fffffcp-127",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_value,
            m_smallest_normalized,
            "0x1.000002p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_value,
            p_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_smallest_value,
            m_inf,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_smallest_value,
            p_zero,
            "-0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_value,
            m_zero,
            "-0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (m_smallest_value, qnan, "-nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(m_smallest_value, snan, "-nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_smallest_value,
            p_normal_value,
            "-0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_smallest_value,
            m_normal_value,
            "0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_smallest_value,
            p_largest_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_smallest_value,
            m_largest_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_smallest_value,
            p_smallest_value,
            "-0x1p-148",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_value,
            m_smallest_value,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_smallest_value,
            p_smallest_normalized,
            "-0x1.000002p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_value,
            m_smallest_normalized,
            "0x1.fffffcp-127",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            p_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_smallest_normalized,
            m_inf,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_smallest_normalized,
            p_zero,
            "0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            m_zero,
            "0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            qnan,
            "-nan",
            Status::OK,
            Category::NaN,
        ),
        /*
// See Note 1.
(p_smallest_normalized, snan, "-nan", Status::INVALID_OP, Category::NaN),
        */
        (
            p_smallest_normalized,
            p_normal_value,
            "-0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            m_normal_value,
            "0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            p_largest_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            m_largest_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            p_smallest_value,
            "0x1.fffffcp-127",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            m_smallest_value,
            "0x1.000002p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            p_smallest_normalized,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_smallest_normalized,
            m_smallest_normalized,
            "0x1p-125",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            p_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_smallest_normalized,
            m_inf,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_smallest_normalized,
            p_zero,
            "-0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            m_zero,
            "-0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            qnan,
            "-nan",
            Status::OK,
            Category::NaN,
        ),
        /*
// See Note 1.
(m_smallest_normalized, snan, "-nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_smallest_normalized,
            p_normal_value,
            "-0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            m_normal_value,
            "0x1p+0",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            p_largest_value,
            "-0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            m_largest_value,
            "0x1.fffffep+127",
            Status::INEXACT,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            p_smallest_value,
            "-0x1.000002p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            m_smallest_value,
            "-0x1.fffffcp-127",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            p_smallest_normalized,
            "-0x1p-125",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            m_smallest_normalized,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
    ];

    for &(x, y, e_result, e_status, e_category) in &special_cases[..] {
        let status;
        let result = unpack!(status=, x - y);
        assert_eq!(status, e_status);
        assert_eq!(result.category(), e_category);
        assert!(result.bitwise_eq(e_result.parse::<Single>().unwrap()));
    }
}

#[test]
fn multiply() {
    // Test Special Cases against each other and normal values.

    // FIXMES/NOTES:
    // 1. Since we perform only default exception handling all operations with
    // signaling NaNs should have a result that is a quiet NaN. Currently they
    // return sNaN.

    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let qnan = Single::NAN;
    let p_normal_value = "0x1p+0".parse::<Single>().unwrap();
    let m_normal_value = "-0x1p+0".parse::<Single>().unwrap();
    let p_largest_value = Single::largest();
    let m_largest_value = -Single::largest();
    let p_smallest_value = Single::SMALLEST;
    let m_smallest_value = -Single::SMALLEST;
    let p_smallest_normalized = Single::smallest_normalized();
    let m_smallest_normalized = -Single::smallest_normalized();

    let overflow_status = Status::OVERFLOW | Status::INEXACT;
    let underflow_status = Status::UNDERFLOW | Status::INEXACT;

    let special_cases = [
        (p_inf, p_inf, "inf", Status::OK, Category::Infinity),
        (p_inf, m_inf, "-inf", Status::OK, Category::Infinity),
        (p_inf, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(p_inf, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (p_inf, p_normal_value, "inf", Status::OK, Category::Infinity),
        (
            p_inf,
            m_normal_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            p_largest_value,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            m_largest_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            p_smallest_value,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            m_smallest_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            p_smallest_normalized,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            m_smallest_normalized,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (m_inf, p_inf, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_inf, "inf", Status::OK, Category::Infinity),
        (m_inf, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(m_inf, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_inf,
            p_normal_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (m_inf, m_normal_value, "inf", Status::OK, Category::Infinity),
        (
            m_inf,
            p_largest_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            m_largest_value,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            p_smallest_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            m_smallest_value,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            p_smallest_normalized,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            m_smallest_normalized,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (p_zero, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (p_zero, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (p_zero, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_zero, "-0x0p+0", Status::OK, Category::Zero),
        (p_zero, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(p_zero, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (p_zero, p_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (
            p_zero,
            m_normal_value,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_zero,
            p_largest_value,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_zero,
            m_largest_value,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_zero,
            p_smallest_value,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_zero,
            m_smallest_value,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_zero,
            p_smallest_normalized,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_zero,
            m_smallest_normalized,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (m_zero, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (m_zero, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (m_zero, p_zero, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(m_zero, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_zero,
            p_normal_value,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (m_zero, m_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (
            m_zero,
            p_largest_value,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_zero,
            m_largest_value,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_zero,
            p_smallest_value,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_zero,
            m_smallest_value,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_zero,
            p_smallest_normalized,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_zero,
            m_smallest_normalized,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (qnan, p_inf, "nan", Status::OK, Category::NaN),
        (qnan, m_inf, "nan", Status::OK, Category::NaN),
        (qnan, p_zero, "nan", Status::OK, Category::NaN),
        (qnan, m_zero, "nan", Status::OK, Category::NaN),
        (qnan, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(qnan, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (qnan, p_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, m_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, p_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_value, "nan", Status::OK, Category::NaN),
        (
            qnan,
            p_smallest_normalized,
            "nan",
            Status::OK,
            Category::NaN,
        ),
        (
            qnan,
            m_smallest_normalized,
            "nan",
            Status::OK,
            Category::NaN,
        ),
        /*
// See Note 1.
(snan, p_inf, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_inf, "nan", Status::INVALID_OP, Category::NaN),
(snan, p_zero, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_zero, "nan", Status::INVALID_OP, Category::NaN),
(snan, qnan, "nan", Status::INVALID_OP, Category::NaN),
(snan, snan, "nan", Status::INVALID_OP, Category::NaN),
(snan, p_normal_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_normal_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, p_largest_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_largest_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, p_smallest_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_smallest_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, p_smallest_normalized, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_smallest_normalized, "nan", Status::INVALID_OP, Category::NaN),
        */
        (p_normal_value, p_inf, "inf", Status::OK, Category::Infinity),
        (
            p_normal_value,
            m_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (p_normal_value, p_zero, "0x0p+0", Status::OK, Category::Zero),
        (
            p_normal_value,
            m_zero,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (p_normal_value, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(p_normal_value, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            p_normal_value,
            p_normal_value,
            "0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            p_normal_value,
            m_normal_value,
            "-0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            p_normal_value,
            p_largest_value,
            "0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            p_normal_value,
            m_largest_value,
            "-0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            p_normal_value,
            p_smallest_value,
            "0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            p_normal_value,
            m_smallest_value,
            "-0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            p_normal_value,
            p_smallest_normalized,
            "0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            p_normal_value,
            m_smallest_normalized,
            "-0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            m_normal_value,
            p_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (m_normal_value, m_inf, "inf", Status::OK, Category::Infinity),
        (
            m_normal_value,
            p_zero,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (m_normal_value, m_zero, "0x0p+0", Status::OK, Category::Zero),
        (m_normal_value, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(m_normal_value, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_normal_value,
            p_normal_value,
            "-0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            m_normal_value,
            m_normal_value,
            "0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            m_normal_value,
            p_largest_value,
            "-0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            m_normal_value,
            m_largest_value,
            "0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            m_normal_value,
            p_smallest_value,
            "-0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            m_normal_value,
            m_smallest_value,
            "0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            m_normal_value,
            p_smallest_normalized,
            "-0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            m_normal_value,
            m_smallest_normalized,
            "0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            p_largest_value,
            p_inf,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_largest_value,
            m_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_largest_value,
            p_zero,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_largest_value,
            m_zero,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (p_largest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(p_largest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            p_largest_value,
            p_normal_value,
            "0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            p_largest_value,
            m_normal_value,
            "-0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            p_largest_value,
            p_largest_value,
            "inf",
            overflow_status,
            Category::Infinity,
        ),
        (
            p_largest_value,
            m_largest_value,
            "-inf",
            overflow_status,
            Category::Infinity,
        ),
        (
            p_largest_value,
            p_smallest_value,
            "0x1.fffffep-22",
            Status::OK,
            Category::Normal,
        ),
        (
            p_largest_value,
            m_smallest_value,
            "-0x1.fffffep-22",
            Status::OK,
            Category::Normal,
        ),
        (
            p_largest_value,
            p_smallest_normalized,
            "0x1.fffffep+1",
            Status::OK,
            Category::Normal,
        ),
        (
            p_largest_value,
            m_smallest_normalized,
            "-0x1.fffffep+1",
            Status::OK,
            Category::Normal,
        ),
        (
            m_largest_value,
            p_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_largest_value,
            m_inf,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_largest_value,
            p_zero,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_largest_value,
            m_zero,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (m_largest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(m_largest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_largest_value,
            p_normal_value,
            "-0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            m_largest_value,
            m_normal_value,
            "0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            m_largest_value,
            p_largest_value,
            "-inf",
            overflow_status,
            Category::Infinity,
        ),
        (
            m_largest_value,
            m_largest_value,
            "inf",
            overflow_status,
            Category::Infinity,
        ),
        (
            m_largest_value,
            p_smallest_value,
            "-0x1.fffffep-22",
            Status::OK,
            Category::Normal,
        ),
        (
            m_largest_value,
            m_smallest_value,
            "0x1.fffffep-22",
            Status::OK,
            Category::Normal,
        ),
        (
            m_largest_value,
            p_smallest_normalized,
            "-0x1.fffffep+1",
            Status::OK,
            Category::Normal,
        ),
        (
            m_largest_value,
            m_smallest_normalized,
            "0x1.fffffep+1",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_value,
            p_inf,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_smallest_value,
            m_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_smallest_value,
            p_zero,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_smallest_value,
            m_zero,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (p_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(p_smallest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            p_smallest_value,
            p_normal_value,
            "0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_value,
            m_normal_value,
            "-0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_value,
            p_largest_value,
            "0x1.fffffep-22",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_value,
            m_largest_value,
            "-0x1.fffffep-22",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_value,
            p_smallest_value,
            "0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            p_smallest_value,
            m_smallest_value,
            "-0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            p_smallest_value,
            p_smallest_normalized,
            "0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            p_smallest_value,
            m_smallest_normalized,
            "-0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            m_smallest_value,
            p_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_smallest_value,
            m_inf,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_smallest_value,
            p_zero,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_smallest_value,
            m_zero,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (m_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(m_smallest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_smallest_value,
            p_normal_value,
            "-0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_value,
            m_normal_value,
            "0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_value,
            p_largest_value,
            "-0x1.fffffep-22",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_value,
            m_largest_value,
            "0x1.fffffep-22",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_value,
            p_smallest_value,
            "-0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            m_smallest_value,
            m_smallest_value,
            "0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            m_smallest_value,
            p_smallest_normalized,
            "-0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            m_smallest_value,
            m_smallest_normalized,
            "0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            p_smallest_normalized,
            p_inf,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_smallest_normalized,
            m_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_smallest_normalized,
            p_zero,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_smallest_normalized,
            m_zero,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_smallest_normalized,
            qnan,
            "nan",
            Status::OK,
            Category::NaN,
        ),
        /*
// See Note 1.
(p_smallest_normalized, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            p_smallest_normalized,
            p_normal_value,
            "0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            m_normal_value,
            "-0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            p_largest_value,
            "0x1.fffffep+1",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            m_largest_value,
            "-0x1.fffffep+1",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            p_smallest_value,
            "0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            p_smallest_normalized,
            m_smallest_value,
            "-0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            p_smallest_normalized,
            p_smallest_normalized,
            "0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            p_smallest_normalized,
            m_smallest_normalized,
            "-0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            m_smallest_normalized,
            p_inf,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_smallest_normalized,
            m_inf,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_smallest_normalized,
            p_zero,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_smallest_normalized,
            m_zero,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_smallest_normalized,
            qnan,
            "nan",
            Status::OK,
            Category::NaN,
        ),
        /*
// See Note 1.
(m_smallest_normalized, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_smallest_normalized,
            p_normal_value,
            "-0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            m_normal_value,
            "0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            p_largest_value,
            "-0x1.fffffep+1",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            m_largest_value,
            "0x1.fffffep+1",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            p_smallest_value,
            "-0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            m_smallest_normalized,
            m_smallest_value,
            "0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            m_smallest_normalized,
            p_smallest_normalized,
            "-0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            m_smallest_normalized,
            m_smallest_normalized,
            "0x0p+0",
            underflow_status,
            Category::Zero,
        ),
    ];

    for &(x, y, e_result, e_status, e_category) in &special_cases[..] {
        let status;
        let result = unpack!(status=, x * y);
        assert_eq!(status, e_status);
        assert_eq!(result.category(), e_category);
        assert!(result.bitwise_eq(e_result.parse::<Single>().unwrap()));
    }
}

#[test]
fn divide() {
    // Test Special Cases against each other and normal values.

    // FIXMES/NOTES:
    // 1. Since we perform only default exception handling all operations with
    // signaling NaNs should have a result that is a quiet NaN. Currently they
    // return sNaN.

    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let qnan = Single::NAN;
    let p_normal_value = "0x1p+0".parse::<Single>().unwrap();
    let m_normal_value = "-0x1p+0".parse::<Single>().unwrap();
    let p_largest_value = Single::largest();
    let m_largest_value = -Single::largest();
    let p_smallest_value = Single::SMALLEST;
    let m_smallest_value = -Single::SMALLEST;
    let p_smallest_normalized = Single::smallest_normalized();
    let m_smallest_normalized = -Single::smallest_normalized();

    let overflow_status = Status::OVERFLOW | Status::INEXACT;
    let underflow_status = Status::UNDERFLOW | Status::INEXACT;

    let special_cases = [
        (p_inf, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (p_inf, p_zero, "inf", Status::OK, Category::Infinity),
        (p_inf, m_zero, "-inf", Status::OK, Category::Infinity),
        (p_inf, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(p_inf, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (p_inf, p_normal_value, "inf", Status::OK, Category::Infinity),
        (
            p_inf,
            m_normal_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            p_largest_value,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            m_largest_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            p_smallest_value,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            m_smallest_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            p_smallest_normalized,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            p_inf,
            m_smallest_normalized,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (m_inf, p_inf, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, m_inf, "nan", Status::INVALID_OP, Category::NaN),
        (m_inf, p_zero, "-inf", Status::OK, Category::Infinity),
        (m_inf, m_zero, "inf", Status::OK, Category::Infinity),
        (m_inf, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(m_inf, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_inf,
            p_normal_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (m_inf, m_normal_value, "inf", Status::OK, Category::Infinity),
        (
            m_inf,
            p_largest_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            m_largest_value,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            p_smallest_value,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            m_smallest_value,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            p_smallest_normalized,
            "-inf",
            Status::OK,
            Category::Infinity,
        ),
        (
            m_inf,
            m_smallest_normalized,
            "inf",
            Status::OK,
            Category::Infinity,
        ),
        (p_zero, p_inf, "0x0p+0", Status::OK, Category::Zero),
        (p_zero, m_inf, "-0x0p+0", Status::OK, Category::Zero),
        (p_zero, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_zero, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (p_zero, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(p_zero, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (p_zero, p_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (
            p_zero,
            m_normal_value,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_zero,
            p_largest_value,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_zero,
            m_largest_value,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_zero,
            p_smallest_value,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_zero,
            m_smallest_value,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_zero,
            p_smallest_normalized,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_zero,
            m_smallest_normalized,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (m_zero, p_inf, "-0x0p+0", Status::OK, Category::Zero),
        (m_zero, m_inf, "0x0p+0", Status::OK, Category::Zero),
        (m_zero, p_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_zero, m_zero, "nan", Status::INVALID_OP, Category::NaN),
        (m_zero, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(m_zero, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_zero,
            p_normal_value,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (m_zero, m_normal_value, "0x0p+0", Status::OK, Category::Zero),
        (
            m_zero,
            p_largest_value,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_zero,
            m_largest_value,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_zero,
            p_smallest_value,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_zero,
            m_smallest_value,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_zero,
            p_smallest_normalized,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_zero,
            m_smallest_normalized,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (qnan, p_inf, "nan", Status::OK, Category::NaN),
        (qnan, m_inf, "nan", Status::OK, Category::NaN),
        (qnan, p_zero, "nan", Status::OK, Category::NaN),
        (qnan, m_zero, "nan", Status::OK, Category::NaN),
        (qnan, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(qnan, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (qnan, p_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, m_normal_value, "nan", Status::OK, Category::NaN),
        (qnan, p_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_largest_value, "nan", Status::OK, Category::NaN),
        (qnan, p_smallest_value, "nan", Status::OK, Category::NaN),
        (qnan, m_smallest_value, "nan", Status::OK, Category::NaN),
        (
            qnan,
            p_smallest_normalized,
            "nan",
            Status::OK,
            Category::NaN,
        ),
        (
            qnan,
            m_smallest_normalized,
            "nan",
            Status::OK,
            Category::NaN,
        ),
        /*
// See Note 1.
(snan, p_inf, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_inf, "nan", Status::INVALID_OP, Category::NaN),
(snan, p_zero, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_zero, "nan", Status::INVALID_OP, Category::NaN),
(snan, qnan, "nan", Status::INVALID_OP, Category::NaN),
(snan, snan, "nan", Status::INVALID_OP, Category::NaN),
(snan, p_normal_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_normal_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, p_largest_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_largest_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, p_smallest_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_smallest_value, "nan", Status::INVALID_OP, Category::NaN),
(snan, p_smallest_normalized, "nan", Status::INVALID_OP, Category::NaN),
(snan, m_smallest_normalized, "nan", Status::INVALID_OP, Category::NaN),
        */
        (p_normal_value, p_inf, "0x0p+0", Status::OK, Category::Zero),
        (p_normal_value, m_inf, "-0x0p+0", Status::OK, Category::Zero),
        (
            p_normal_value,
            p_zero,
            "inf",
            Status::DIV_BY_ZERO,
            Category::Infinity,
        ),
        (
            p_normal_value,
            m_zero,
            "-inf",
            Status::DIV_BY_ZERO,
            Category::Infinity,
        ),
        (p_normal_value, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(p_normal_value, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            p_normal_value,
            p_normal_value,
            "0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            p_normal_value,
            m_normal_value,
            "-0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            p_normal_value,
            p_largest_value,
            "0x1p-128",
            underflow_status,
            Category::Normal,
        ),
        (
            p_normal_value,
            m_largest_value,
            "-0x1p-128",
            underflow_status,
            Category::Normal,
        ),
        (
            p_normal_value,
            p_smallest_value,
            "inf",
            overflow_status,
            Category::Infinity,
        ),
        (
            p_normal_value,
            m_smallest_value,
            "-inf",
            overflow_status,
            Category::Infinity,
        ),
        (
            p_normal_value,
            p_smallest_normalized,
            "0x1p+126",
            Status::OK,
            Category::Normal,
        ),
        (
            p_normal_value,
            m_smallest_normalized,
            "-0x1p+126",
            Status::OK,
            Category::Normal,
        ),
        (m_normal_value, p_inf, "-0x0p+0", Status::OK, Category::Zero),
        (m_normal_value, m_inf, "0x0p+0", Status::OK, Category::Zero),
        (
            m_normal_value,
            p_zero,
            "-inf",
            Status::DIV_BY_ZERO,
            Category::Infinity,
        ),
        (
            m_normal_value,
            m_zero,
            "inf",
            Status::DIV_BY_ZERO,
            Category::Infinity,
        ),
        (m_normal_value, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(m_normal_value, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_normal_value,
            p_normal_value,
            "-0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            m_normal_value,
            m_normal_value,
            "0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            m_normal_value,
            p_largest_value,
            "-0x1p-128",
            underflow_status,
            Category::Normal,
        ),
        (
            m_normal_value,
            m_largest_value,
            "0x1p-128",
            underflow_status,
            Category::Normal,
        ),
        (
            m_normal_value,
            p_smallest_value,
            "-inf",
            overflow_status,
            Category::Infinity,
        ),
        (
            m_normal_value,
            m_smallest_value,
            "inf",
            overflow_status,
            Category::Infinity,
        ),
        (
            m_normal_value,
            p_smallest_normalized,
            "-0x1p+126",
            Status::OK,
            Category::Normal,
        ),
        (
            m_normal_value,
            m_smallest_normalized,
            "0x1p+126",
            Status::OK,
            Category::Normal,
        ),
        (p_largest_value, p_inf, "0x0p+0", Status::OK, Category::Zero),
        (
            p_largest_value,
            m_inf,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_largest_value,
            p_zero,
            "inf",
            Status::DIV_BY_ZERO,
            Category::Infinity,
        ),
        (
            p_largest_value,
            m_zero,
            "-inf",
            Status::DIV_BY_ZERO,
            Category::Infinity,
        ),
        (p_largest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(p_largest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            p_largest_value,
            p_normal_value,
            "0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            p_largest_value,
            m_normal_value,
            "-0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            p_largest_value,
            p_largest_value,
            "0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            p_largest_value,
            m_largest_value,
            "-0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            p_largest_value,
            p_smallest_value,
            "inf",
            overflow_status,
            Category::Infinity,
        ),
        (
            p_largest_value,
            m_smallest_value,
            "-inf",
            overflow_status,
            Category::Infinity,
        ),
        (
            p_largest_value,
            p_smallest_normalized,
            "inf",
            overflow_status,
            Category::Infinity,
        ),
        (
            p_largest_value,
            m_smallest_normalized,
            "-inf",
            overflow_status,
            Category::Infinity,
        ),
        (
            m_largest_value,
            p_inf,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (m_largest_value, m_inf, "0x0p+0", Status::OK, Category::Zero),
        (
            m_largest_value,
            p_zero,
            "-inf",
            Status::DIV_BY_ZERO,
            Category::Infinity,
        ),
        (
            m_largest_value,
            m_zero,
            "inf",
            Status::DIV_BY_ZERO,
            Category::Infinity,
        ),
        (m_largest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(m_largest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_largest_value,
            p_normal_value,
            "-0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            m_largest_value,
            m_normal_value,
            "0x1.fffffep+127",
            Status::OK,
            Category::Normal,
        ),
        (
            m_largest_value,
            p_largest_value,
            "-0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            m_largest_value,
            m_largest_value,
            "0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            m_largest_value,
            p_smallest_value,
            "-inf",
            overflow_status,
            Category::Infinity,
        ),
        (
            m_largest_value,
            m_smallest_value,
            "inf",
            overflow_status,
            Category::Infinity,
        ),
        (
            m_largest_value,
            p_smallest_normalized,
            "-inf",
            overflow_status,
            Category::Infinity,
        ),
        (
            m_largest_value,
            m_smallest_normalized,
            "inf",
            overflow_status,
            Category::Infinity,
        ),
        (
            p_smallest_value,
            p_inf,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_smallest_value,
            m_inf,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_smallest_value,
            p_zero,
            "inf",
            Status::DIV_BY_ZERO,
            Category::Infinity,
        ),
        (
            p_smallest_value,
            m_zero,
            "-inf",
            Status::DIV_BY_ZERO,
            Category::Infinity,
        ),
        (p_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(p_smallest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            p_smallest_value,
            p_normal_value,
            "0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_value,
            m_normal_value,
            "-0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_value,
            p_largest_value,
            "0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            p_smallest_value,
            m_largest_value,
            "-0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            p_smallest_value,
            p_smallest_value,
            "0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_value,
            m_smallest_value,
            "-0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_value,
            p_smallest_normalized,
            "0x1p-23",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_value,
            m_smallest_normalized,
            "-0x1p-23",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_value,
            p_inf,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_smallest_value,
            m_inf,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_smallest_value,
            p_zero,
            "-inf",
            Status::DIV_BY_ZERO,
            Category::Infinity,
        ),
        (
            m_smallest_value,
            m_zero,
            "inf",
            Status::DIV_BY_ZERO,
            Category::Infinity,
        ),
        (m_smallest_value, qnan, "nan", Status::OK, Category::NaN),
        /*
// See Note 1.
(m_smallest_value, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_smallest_value,
            p_normal_value,
            "-0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_value,
            m_normal_value,
            "0x1p-149",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_value,
            p_largest_value,
            "-0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            m_smallest_value,
            m_largest_value,
            "0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            m_smallest_value,
            p_smallest_value,
            "-0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_value,
            m_smallest_value,
            "0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_value,
            p_smallest_normalized,
            "-0x1p-23",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_value,
            m_smallest_normalized,
            "0x1p-23",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            p_inf,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_smallest_normalized,
            m_inf,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            p_smallest_normalized,
            p_zero,
            "inf",
            Status::DIV_BY_ZERO,
            Category::Infinity,
        ),
        (
            p_smallest_normalized,
            m_zero,
            "-inf",
            Status::DIV_BY_ZERO,
            Category::Infinity,
        ),
        (
            p_smallest_normalized,
            qnan,
            "nan",
            Status::OK,
            Category::NaN,
        ),
        /*
// See Note 1.
(p_smallest_normalized, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            p_smallest_normalized,
            p_normal_value,
            "0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            m_normal_value,
            "-0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            p_largest_value,
            "0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            p_smallest_normalized,
            m_largest_value,
            "-0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            p_smallest_normalized,
            p_smallest_value,
            "0x1p+23",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            m_smallest_value,
            "-0x1p+23",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            p_smallest_normalized,
            "0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            p_smallest_normalized,
            m_smallest_normalized,
            "-0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            p_inf,
            "-0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_smallest_normalized,
            m_inf,
            "0x0p+0",
            Status::OK,
            Category::Zero,
        ),
        (
            m_smallest_normalized,
            p_zero,
            "-inf",
            Status::DIV_BY_ZERO,
            Category::Infinity,
        ),
        (
            m_smallest_normalized,
            m_zero,
            "inf",
            Status::DIV_BY_ZERO,
            Category::Infinity,
        ),
        (
            m_smallest_normalized,
            qnan,
            "nan",
            Status::OK,
            Category::NaN,
        ),
        /*
// See Note 1.
(m_smallest_normalized, snan, "nan", Status::INVALID_OP, Category::NaN),
        */
        (
            m_smallest_normalized,
            p_normal_value,
            "-0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            m_normal_value,
            "0x1p-126",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            p_largest_value,
            "-0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            m_smallest_normalized,
            m_largest_value,
            "0x0p+0",
            underflow_status,
            Category::Zero,
        ),
        (
            m_smallest_normalized,
            p_smallest_value,
            "-0x1p+23",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            m_smallest_value,
            "0x1p+23",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            p_smallest_normalized,
            "-0x1p+0",
            Status::OK,
            Category::Normal,
        ),
        (
            m_smallest_normalized,
            m_smallest_normalized,
            "0x1p+0",
            Status::OK,
            Category::Normal,
        ),
    ];

    for &(x, y, e_result, e_status, e_category) in &special_cases[..] {
        let status;
        let result = unpack!(status=, x / y);
        assert_eq!(status, e_status);
        assert_eq!(result.category(), e_category);
        assert!(result.bitwise_eq(e_result.parse::<Single>().unwrap()));
    }
}

#[test]
fn operator_overloads() {
    // This is mostly testing that these operator overloads compile.
    let one = "0x1p+0".parse::<Single>().unwrap();
    let two = "0x2p+0".parse::<Single>().unwrap();
    assert!(two.bitwise_eq((one + one).value));
    assert!(one.bitwise_eq((two - one).value));
    assert!(two.bitwise_eq((one * two).value));
    assert!(one.bitwise_eq((two / two).value));
}

#[test]
fn abs() {
    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let p_qnan = Single::NAN;
    let m_qnan = -Single::NAN;
    let p_snan = Single::snan(None);
    let m_snan = -Single::snan(None);
    let p_normal_value = "0x1p+0".parse::<Single>().unwrap();
    let m_normal_value = "-0x1p+0".parse::<Single>().unwrap();
    let p_largest_value = Single::largest();
    let m_largest_value = -Single::largest();
    let p_smallest_value = Single::SMALLEST;
    let m_smallest_value = -Single::SMALLEST;
    let p_smallest_normalized = Single::smallest_normalized();
    let m_smallest_normalized = -Single::smallest_normalized();

    assert!(p_inf.bitwise_eq(p_inf.abs()));
    assert!(p_inf.bitwise_eq(m_inf.abs()));
    assert!(p_zero.bitwise_eq(p_zero.abs()));
    assert!(p_zero.bitwise_eq(m_zero.abs()));
    assert!(p_qnan.bitwise_eq(p_qnan.abs()));
    assert!(p_qnan.bitwise_eq(m_qnan.abs()));
    assert!(p_snan.bitwise_eq(p_snan.abs()));
    assert!(p_snan.bitwise_eq(m_snan.abs()));
    assert!(p_normal_value.bitwise_eq(p_normal_value.abs()));
    assert!(p_normal_value.bitwise_eq(m_normal_value.abs()));
    assert!(p_largest_value.bitwise_eq(p_largest_value.abs()));
    assert!(p_largest_value.bitwise_eq(m_largest_value.abs()));
    assert!(p_smallest_value.bitwise_eq(p_smallest_value.abs()));
    assert!(p_smallest_value.bitwise_eq(m_smallest_value.abs()));
    assert!(p_smallest_normalized.bitwise_eq(
        p_smallest_normalized.abs(),
    ));
    assert!(p_smallest_normalized.bitwise_eq(
        m_smallest_normalized.abs(),
    ));
}

#[test]
fn neg() {
    let one = "1.0".parse::<Single>().unwrap();
    let neg_one = "-1.0".parse::<Single>().unwrap();
    let zero = Single::ZERO;
    let neg_zero = -Single::ZERO;
    let inf = Single::INFINITY;
    let neg_inf = -Single::INFINITY;
    let qnan = Single::NAN;
    let neg_qnan = -Single::NAN;

    assert!(neg_one.bitwise_eq(-one));
    assert!(one.bitwise_eq(-neg_one));
    assert!(neg_zero.bitwise_eq(-zero));
    assert!(zero.bitwise_eq(-neg_zero));
    assert!(neg_inf.bitwise_eq(-inf));
    assert!(inf.bitwise_eq(-neg_inf));
    assert!(neg_inf.bitwise_eq(-inf));
    assert!(inf.bitwise_eq(-neg_inf));
    assert!(neg_qnan.bitwise_eq(-qnan));
    assert!(qnan.bitwise_eq(-neg_qnan));
}

#[test]
fn ilogb() {
    assert_eq!(-1074, Double::SMALLEST.ilogb());
    assert_eq!(-1074, (-Double::SMALLEST).ilogb());
    assert_eq!(
        -1023,
        "0x1.ffffffffffffep-1024".parse::<Double>().unwrap().ilogb()
    );
    assert_eq!(
        -1023,
        "0x1.ffffffffffffep-1023".parse::<Double>().unwrap().ilogb()
    );
    assert_eq!(
        -1023,
        "-0x1.ffffffffffffep-1023"
            .parse::<Double>()
            .unwrap()
            .ilogb()
    );
    assert_eq!(-51, "0x1p-51".parse::<Double>().unwrap().ilogb());
    assert_eq!(
        -1023,
        "0x1.c60f120d9f87cp-1023".parse::<Double>().unwrap().ilogb()
    );
    assert_eq!(-2, "0x0.ffffp-1".parse::<Double>().unwrap().ilogb());
    assert_eq!(-1023, "0x1.fffep-1023".parse::<Double>().unwrap().ilogb());
    assert_eq!(1023, Double::largest().ilogb());
    assert_eq!(1023, (-Double::largest()).ilogb());


    assert_eq!(0, "0x1p+0".parse::<Single>().unwrap().ilogb());
    assert_eq!(0, "-0x1p+0".parse::<Single>().unwrap().ilogb());
    assert_eq!(42, "0x1p+42".parse::<Single>().unwrap().ilogb());
    assert_eq!(-42, "0x1p-42".parse::<Single>().unwrap().ilogb());

    assert_eq!(IEK_INF, Single::INFINITY.ilogb());
    assert_eq!(IEK_INF, (-Single::INFINITY).ilogb());
    assert_eq!(IEK_ZERO, Single::ZERO.ilogb());
    assert_eq!(IEK_ZERO, (-Single::ZERO).ilogb());
    assert_eq!(IEK_NAN, Single::NAN.ilogb());
    assert_eq!(IEK_NAN, Single::snan(None).ilogb());

    assert_eq!(127, Single::largest().ilogb());
    assert_eq!(127, (-Single::largest()).ilogb());

    assert_eq!(-149, Single::SMALLEST.ilogb());
    assert_eq!(-149, (-Single::SMALLEST).ilogb());
    assert_eq!(-126, Single::smallest_normalized().ilogb());
    assert_eq!(-126, (-Single::smallest_normalized()).ilogb());
}

#[test]
fn scalbn() {
    assert!("0x1p+0".parse::<Single>().unwrap().bitwise_eq(
        "0x1p+0".parse::<Single>().unwrap().scalbn(0),
    ));
    assert!("0x1p+42".parse::<Single>().unwrap().bitwise_eq(
        "0x1p+0".parse::<Single>().unwrap().scalbn(42),
    ));
    assert!("0x1p-42".parse::<Single>().unwrap().bitwise_eq(
        "0x1p+0".parse::<Single>().unwrap().scalbn(-42),
    ));

    let p_inf = Single::INFINITY;
    let m_inf = -Single::INFINITY;
    let p_zero = Single::ZERO;
    let m_zero = -Single::ZERO;
    let p_qnan = Single::NAN;
    let m_qnan = -Single::NAN;
    let snan = Single::snan(None);

    assert!(p_inf.bitwise_eq(p_inf.scalbn(0)));
    assert!(m_inf.bitwise_eq(m_inf.scalbn(0)));
    assert!(p_zero.bitwise_eq(p_zero.scalbn(0)));
    assert!(m_zero.bitwise_eq(m_zero.scalbn(0)));
    assert!(p_qnan.bitwise_eq(p_qnan.scalbn(0)));
    assert!(m_qnan.bitwise_eq(m_qnan.scalbn(0)));
    assert!(!snan.scalbn(0).is_signaling());

    let scalbn_snan = snan.scalbn(1);
    assert!(scalbn_snan.is_nan() && !scalbn_snan.is_signaling());

    // Make sure highest bit of payload is preserved.
    let payload = (1 << 50) | (1 << 49) | (1234 << 32) | 1;

    let snan_with_payload = Double::snan(Some(payload));
    let quiet_payload = snan_with_payload.scalbn(1);
    assert!(quiet_payload.is_nan() && !quiet_payload.is_signaling());
    assert_eq!(payload, quiet_payload.to_bits() & ((1 << 51) - 1));

    assert!(p_inf.bitwise_eq(
        "0x1p+0".parse::<Single>().unwrap().scalbn(128),
    ));
    assert!(m_inf.bitwise_eq(
        "-0x1p+0".parse::<Single>().unwrap().scalbn(128),
    ));
    assert!(p_inf.bitwise_eq(
        "0x1p+127".parse::<Single>().unwrap().scalbn(1),
    ));
    assert!(p_zero.bitwise_eq(
        "0x1p-127".parse::<Single>().unwrap().scalbn(-127),
    ));
    assert!(m_zero.bitwise_eq(
        "-0x1p-127".parse::<Single>().unwrap().scalbn(-127),
    ));
    assert!("-0x1p-149".parse::<Single>().unwrap().bitwise_eq(
        "-0x1p-127".parse::<Single>().unwrap().scalbn(-22),
    ));
    assert!(p_zero.bitwise_eq(
        "0x1p-126".parse::<Single>().unwrap().scalbn(-24),
    ));


    let smallest_f64 = Double::SMALLEST;
    let neg_smallest_f64 = -Double::SMALLEST;

    let largest_f64 = Double::largest();
    let neg_largest_f64 = -Double::largest();

    let largest_denormal_f64 = "0x1.ffffffffffffep-1023".parse::<Double>().unwrap();
    let neg_largest_denormal_f64 = "-0x1.ffffffffffffep-1023".parse::<Double>().unwrap();


    assert!(smallest_f64.bitwise_eq(
        "0x1p-1074".parse::<Double>().unwrap().scalbn(0),
    ));
    assert!(neg_smallest_f64.bitwise_eq(
        "-0x1p-1074".parse::<Double>().unwrap().scalbn(0),
    ));

    assert!("0x1p+1023".parse::<Double>().unwrap().bitwise_eq(
        smallest_f64.scalbn(
            2097,
        ),
    ));

    assert!(smallest_f64.scalbn(-2097).is_pos_zero());
    assert!(smallest_f64.scalbn(-2098).is_pos_zero());
    assert!(smallest_f64.scalbn(-2099).is_pos_zero());
    assert!("0x1p+1022".parse::<Double>().unwrap().bitwise_eq(
        smallest_f64.scalbn(
            2096,
        ),
    ));
    assert!("0x1p+1023".parse::<Double>().unwrap().bitwise_eq(
        smallest_f64.scalbn(
            2097,
        ),
    ));
    assert!(smallest_f64.scalbn(2098).is_infinite());
    assert!(smallest_f64.scalbn(2099).is_infinite());

    // Test for integer overflows when adding to exponent.
    assert!(smallest_f64.scalbn(-ExpInt::max_value()).is_pos_zero());
    assert!(largest_f64.scalbn(ExpInt::max_value()).is_infinite());

    assert!(largest_denormal_f64.bitwise_eq(
        largest_denormal_f64.scalbn(0),
    ));
    assert!(neg_largest_denormal_f64.bitwise_eq(
        neg_largest_denormal_f64.scalbn(0),
    ));

    assert!(
        "0x1.ffffffffffffep-1022"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(largest_denormal_f64.scalbn(1))
    );
    assert!(
        "-0x1.ffffffffffffep-1021"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(neg_largest_denormal_f64.scalbn(2))
    );

    assert!(
        "0x1.ffffffffffffep+1"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(largest_denormal_f64.scalbn(1024))
    );
    assert!(largest_denormal_f64.scalbn(-1023).is_pos_zero());
    assert!(largest_denormal_f64.scalbn(-1024).is_pos_zero());
    assert!(largest_denormal_f64.scalbn(-2048).is_pos_zero());
    assert!(largest_denormal_f64.scalbn(2047).is_infinite());
    assert!(largest_denormal_f64.scalbn(2098).is_infinite());
    assert!(largest_denormal_f64.scalbn(2099).is_infinite());

    assert!(
        "0x1.ffffffffffffep-2"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(largest_denormal_f64.scalbn(1021))
    );
    assert!(
        "0x1.ffffffffffffep-1"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(largest_denormal_f64.scalbn(1022))
    );
    assert!(
        "0x1.ffffffffffffep+0"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(largest_denormal_f64.scalbn(1023))
    );
    assert!(
        "0x1.ffffffffffffep+1023"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(largest_denormal_f64.scalbn(2046))
    );
    assert!("0x1p+974".parse::<Double>().unwrap().bitwise_eq(
        smallest_f64.scalbn(
            2048,
        ),
    ));

    let random_denormal_f64 = "0x1.c60f120d9f87cp+51".parse::<Double>().unwrap();
    assert!(
        "0x1.c60f120d9f87cp-972"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(random_denormal_f64.scalbn(-1023))
    );
    assert!(
        "0x1.c60f120d9f87cp-1"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(random_denormal_f64.scalbn(-52))
    );
    assert!(
        "0x1.c60f120d9f87cp-2"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(random_denormal_f64.scalbn(-53))
    );
    assert!(
        "0x1.c60f120d9f87cp+0"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(random_denormal_f64.scalbn(-51))
    );

    assert!(random_denormal_f64.scalbn(-2097).is_pos_zero());
    assert!(random_denormal_f64.scalbn(-2090).is_pos_zero());


    assert!("-0x1p-1073".parse::<Double>().unwrap().bitwise_eq(
        neg_largest_f64.scalbn(-2097),
    ));

    assert!("-0x1p-1024".parse::<Double>().unwrap().bitwise_eq(
        neg_largest_f64.scalbn(-2048),
    ));

    assert!("0x1p-1073".parse::<Double>().unwrap().bitwise_eq(
        largest_f64.scalbn(
            -2097,
        ),
    ));

    assert!("0x1p-1074".parse::<Double>().unwrap().bitwise_eq(
        largest_f64.scalbn(
            -2098,
        ),
    ));
    assert!("-0x1p-1074".parse::<Double>().unwrap().bitwise_eq(
        neg_largest_f64.scalbn(-2098),
    ));
    assert!(neg_largest_f64.scalbn(-2099).is_neg_zero());
    assert!(largest_f64.scalbn(1).is_infinite());


    assert!("0x1p+0".parse::<Double>().unwrap().bitwise_eq(
        "0x1p+52".parse::<Double>().unwrap().scalbn(-52),
    ));

    assert!("0x1p-103".parse::<Double>().unwrap().bitwise_eq(
        "0x1p-51".parse::<Double>().unwrap().scalbn(-52),
    ));
}

#[test]
fn frexp() {
    let p_zero = Double::ZERO;
    let m_zero = -Double::ZERO;
    let one = Double::from_f64(1.0);
    let m_one = Double::from_f64(-1.0);

    let largest_denormal = "0x1.ffffffffffffep-1023".parse::<Double>().unwrap();
    let neg_largest_denormal = "-0x1.ffffffffffffep-1023".parse::<Double>().unwrap();

    let smallest = Double::SMALLEST;
    let neg_smallest = -Double::SMALLEST;

    let largest = Double::largest();
    let neg_largest = -Double::largest();

    let p_inf = Double::INFINITY;
    let m_inf = -Double::INFINITY;

    let p_qnan = Double::NAN;
    let m_qnan = -Double::NAN;
    let snan = Double::snan(None);

    // Make sure highest bit of payload is preserved.
    let payload = (1 << 50) | (1 << 49) | (1234 << 32) | 1;

    let snan_with_payload = Double::snan(Some(payload));

    let mut exp = 0;

    let frac = p_zero.frexp(&mut exp);
    assert_eq!(0, exp);
    assert!(frac.is_pos_zero());

    let frac = m_zero.frexp(&mut exp);
    assert_eq!(0, exp);
    assert!(frac.is_neg_zero());


    let frac = one.frexp(&mut exp);
    assert_eq!(1, exp);
    assert!("0x1p-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = m_one.frexp(&mut exp);
    assert_eq!(1, exp);
    assert!("-0x1p-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = largest_denormal.frexp(&mut exp);
    assert_eq!(-1022, exp);
    assert!(
        "0x1.ffffffffffffep-1"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(frac)
    );

    let frac = neg_largest_denormal.frexp(&mut exp);
    assert_eq!(-1022, exp);
    assert!(
        "-0x1.ffffffffffffep-1"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(frac)
    );


    let frac = smallest.frexp(&mut exp);
    assert_eq!(-1073, exp);
    assert!("0x1p-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = neg_smallest.frexp(&mut exp);
    assert_eq!(-1073, exp);
    assert!("-0x1p-1".parse::<Double>().unwrap().bitwise_eq(frac));


    let frac = largest.frexp(&mut exp);
    assert_eq!(1024, exp);
    assert!(
        "0x1.fffffffffffffp-1"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(frac)
    );

    let frac = neg_largest.frexp(&mut exp);
    assert_eq!(1024, exp);
    assert!(
        "-0x1.fffffffffffffp-1"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(frac)
    );


    let frac = p_inf.frexp(&mut exp);
    assert_eq!(IEK_INF, exp);
    assert!(frac.is_infinite() && !frac.is_negative());

    let frac = m_inf.frexp(&mut exp);
    assert_eq!(IEK_INF, exp);
    assert!(frac.is_infinite() && frac.is_negative());

    let frac = p_qnan.frexp(&mut exp);
    assert_eq!(IEK_NAN, exp);
    assert!(frac.is_nan());

    let frac = m_qnan.frexp(&mut exp);
    assert_eq!(IEK_NAN, exp);
    assert!(frac.is_nan());

    let frac = snan.frexp(&mut exp);
    assert_eq!(IEK_NAN, exp);
    assert!(frac.is_nan() && !frac.is_signaling());

    let frac = snan_with_payload.frexp(&mut exp);
    assert_eq!(IEK_NAN, exp);
    assert!(frac.is_nan() && !frac.is_signaling());
    assert_eq!(payload, frac.to_bits() & ((1 << 51) - 1));

    let frac = "0x0.ffffp-1".parse::<Double>().unwrap().frexp(&mut exp);
    assert_eq!(-1, exp);
    assert!("0x1.fffep-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = "0x1p-51".parse::<Double>().unwrap().frexp(&mut exp);
    assert_eq!(-50, exp);
    assert!("0x1p-1".parse::<Double>().unwrap().bitwise_eq(frac));

    let frac = "0x1.c60f120d9f87cp+51".parse::<Double>().unwrap().frexp(
        &mut exp,
    );
    assert_eq!(52, exp);
    assert!(
        "0x1.c60f120d9f87cp-1"
            .parse::<Double>()
            .unwrap()
            .bitwise_eq(frac)
    );
}

#[test]
fn modulo() {
    let mut status;
    {
        let f1 = "1.5".parse::<Double>().unwrap();
        let f2 = "1.0".parse::<Double>().unwrap();
        let expected = "0.5".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "0.5".parse::<Double>().unwrap();
        let f2 = "1.0".parse::<Double>().unwrap();
        let expected = "0.5".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "0x1.3333333333333p-2".parse::<Double>().unwrap(); // 0.3
        let f2 = "0x1.47ae147ae147bp-7".parse::<Double>().unwrap(); // 0.01
        // 0.009999999999999983
        let expected = "0x1.47ae147ae1471p-7".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "0x1p64".parse::<Double>().unwrap(); // 1.8446744073709552e19
        let f2 = "1.5".parse::<Double>().unwrap();
        let expected = "1.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "0x1p1000".parse::<Double>().unwrap();
        let f2 = "0x1p-1000".parse::<Double>().unwrap();
        let expected = "0.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "0.0".parse::<Double>().unwrap();
        let f2 = "1.0".parse::<Double>().unwrap();
        let expected = "0.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).bitwise_eq(expected));
        assert_eq!(status, Status::OK);
    }
    {
        let f1 = "1.0".parse::<Double>().unwrap();
        let f2 = "0.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).is_nan());
        assert_eq!(status, Status::INVALID_OP);
    }
    {
        let f1 = "0.0".parse::<Double>().unwrap();
        let f2 = "0.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).is_nan());
        assert_eq!(status, Status::INVALID_OP);
    }
    {
        let f1 = Double::INFINITY;
        let f2 = "1.0".parse::<Double>().unwrap();
        assert!(unpack!(status=, f1 % f2).is_nan());
        assert_eq!(status, Status::INVALID_OP);
    }
}
