use rustc_apfloat::ieee::Quad;
use rustc_apfloat::unpack;
use rustc_apfloat::{Round, Status};

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
    let expected = "0x1.fffffffffffffffffffffffffffep+16383".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_infinite() && !test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(-largest) = -largest + inc.
    let test = unpack!(status=, (-Quad::largest()).next_up());
    let expected = "-0x1.fffffffffffffffffffffffffffep+16383".parse::<Quad>().unwrap();
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
    let expected = "0x0.0000000000000000000000000002p-16382".parse::<Quad>().unwrap();
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
    let expected = "-0x0.0000000000000000000000000002p-16382".parse::<Quad>().unwrap();
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
    let expected = "0x1.0000000000000000000000000000p-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(test.bitwise_eq(expected));

    // nextDown(-Largest Denormal) -> -Smallest Normal.
    let test = unpack!(status=, "-0x0.ffffffffffffffffffffffffffffp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "-0x1.0000000000000000000000000000p-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(test.bitwise_eq(expected));

    // nextUp(-Smallest Normal) -> -Largest Denormal.
    let test = unpack!(status=, "-0x1.0000000000000000000000000000p-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "-0x0.ffffffffffffffffffffffffffffp-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(test.bitwise_eq(expected));

    // nextDown(+Smallest Normal) -> +Largest Denormal.
    let test = unpack!(status=, "+0x1.0000000000000000000000000000p-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "+0x0.ffffffffffffffffffffffffffffp-16382".parse::<Quad>().unwrap();
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
    let expected = "-0x1.ffffffffffffffffffffffffffffp+0".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.bitwise_eq(expected));

    // nextDown(+Normal Binade Boundary) -> +Normal Binade Boundary - 1.
    let test = unpack!(status=, "0x1p+1".parse::<Quad>().unwrap().next_down());
    let expected = "0x1.ffffffffffffffffffffffffffffp+0".parse::<Quad>().unwrap();
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
    let expected = "-0x0.fffffffffffffffffffffffffffep-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(+Largest Denormal) -> +Largest Denormal - inc.
    let test = unpack!(status=, "0x0.ffffffffffffffffffffffffffffp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "0x0.fffffffffffffffffffffffffffep-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(+Smallest Normal) -> +Smallest Normal + inc.
    let test = unpack!(status=, "0x1.0000000000000000000000000000p-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "0x1.0000000000000000000000000001p-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(-Smallest Normal) -> -Smallest Normal - inc.
    let test = unpack!(status=, "-0x1.0000000000000000000000000000p-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "-0x1.0000000000000000000000000001p-16382".parse::<Quad>().unwrap();
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
    let expected = "-0x1.ffffffffffffffffffffffffffffp-16382".parse::<Quad>().unwrap();
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
    let expected = "0x1.ffffffffffffffffffffffffffffp-16382".parse::<Quad>().unwrap();
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
    let expected = "0x0.ffffffffffffffffffffffff000dp-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(+Denormal) -> +Denormal.
    let test = unpack!(status=, "0x0.ffffffffffffffffffffffff000cp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "0x0.ffffffffffffffffffffffff000bp-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(-Denormal) -> -Denormal.
    let test = unpack!(status=, "-0x0.ffffffffffffffffffffffff000cp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "-0x0.ffffffffffffffffffffffff000bp-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(-Denormal) -> -Denormal
    let test = unpack!(status=, "-0x0.ffffffffffffffffffffffff000cp-16382"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "-0x0.ffffffffffffffffffffffff000dp-16382".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(test.is_denormal());
    assert!(test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(+Normal) -> +Normal.
    let test = unpack!(status=, "0x1.ffffffffffffffffffffffff000cp-16000"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "0x1.ffffffffffffffffffffffff000dp-16000".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(+Normal) -> +Normal.
    let test = unpack!(status=, "0x1.ffffffffffffffffffffffff000cp-16000"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "0x1.ffffffffffffffffffffffff000bp-16000".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(!test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextUp(-Normal) -> -Normal.
    let test = unpack!(status=, "-0x1.ffffffffffffffffffffffff000cp-16000"
        .parse::<Quad>()
        .unwrap()
        .next_up());
    let expected = "-0x1.ffffffffffffffffffffffff000bp-16000".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(test.is_negative());
    assert!(test.bitwise_eq(expected));

    // nextDown(-Normal) -> -Normal.
    let test = unpack!(status=, "-0x1.ffffffffffffffffffffffff000cp-16000"
        .parse::<Quad>()
        .unwrap()
        .next_down());
    let expected = "-0x1.ffffffffffffffffffffffff000dp-16000".parse::<Quad>().unwrap();
    assert_eq!(status, Status::OK);
    assert!(!test.is_denormal());
    assert!(test.is_negative());
    assert!(test.bitwise_eq(expected));
}
