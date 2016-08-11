use std::panic;

use quickcheck::TestResult;

quickcheck! {
    fn udivmoddi4(n: (u32, u32), d: (u32, u32)) -> TestResult {
        let n = ::U64 { low: n.0, high: n.1 }[..];
        let d = ::U64 { low: d.0, high: d.1 }[..];

        if d == 0 {
            TestResult::discard()
        } else {
            let mut r = 0;
            let q = ::div::__udivmoddi4(n, d, Some(&mut r));

            TestResult::from_bool(q * d + r == n)
        }
    }
}

quickcheck! {
    fn udivmodsi4(n: u32, d: u32) -> TestResult {
        if d == 0 {
            TestResult::discard()
        } else {
            let mut r = 0;
            let q = ::div::__udivmodsi4(n, d, Some(&mut r));

            TestResult::from_bool(q * d + r == n)
        }
    }
}
