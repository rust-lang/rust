use regex::internal::ExecBuilder;

/// Given a regex, check if all of the backends produce the same
/// results on a number of different inputs.
///
/// For now this just throws quickcheck at the problem, which
/// is not very good because it only really tests half of the
/// problem space. It is pretty unlikely that a random string
/// will match any given regex, so this will probably just
/// be checking that the different backends fail in the same
/// way. This is still worthwhile to test, but is definitely not
/// the whole story.
///
/// TODO(ethan): In order to cover the other half of the problem
/// space, we should generate a random matching string by inspecting
/// the AST of the input regex. The right way to do this probably
/// involves adding a custom Arbitrary instance around a couple
/// of newtypes. That way we can respect the quickcheck size hinting
/// and shrinking and whatnot.
pub fn backends_are_consistent(re: &str) -> Result<u64, String> {
    let standard_backends = vec![
        (
            "bounded_backtracking_re",
            ExecBuilder::new(re)
                .bounded_backtracking()
                .build()
                .map(|exec| exec.into_regex())
                .map_err(|err| format!("{}", err))?,
        ),
        (
            "pikevm_re",
            ExecBuilder::new(re)
                .nfa()
                .build()
                .map(|exec| exec.into_regex())
                .map_err(|err| format!("{}", err))?,
        ),
        (
            "default_re",
            ExecBuilder::new(re)
                .build()
                .map(|exec| exec.into_regex())
                .map_err(|err| format!("{}", err))?,
        ),
    ];

    let utf8bytes_backends = vec![
        (
            "bounded_backtracking_utf8bytes_re",
            ExecBuilder::new(re)
                .bounded_backtracking()
                .bytes(true)
                .build()
                .map(|exec| exec.into_regex())
                .map_err(|err| format!("{}", err))?,
        ),
        (
            "pikevm_utf8bytes_re",
            ExecBuilder::new(re)
                .nfa()
                .bytes(true)
                .build()
                .map(|exec| exec.into_regex())
                .map_err(|err| format!("{}", err))?,
        ),
        (
            "default_utf8bytes_re",
            ExecBuilder::new(re)
                .bytes(true)
                .build()
                .map(|exec| exec.into_regex())
                .map_err(|err| format!("{}", err))?,
        ),
    ];

    let bytes_backends = vec![
        (
            "bounded_backtracking_bytes_re",
            ExecBuilder::new(re)
                .bounded_backtracking()
                .only_utf8(false)
                .build()
                .map(|exec| exec.into_byte_regex())
                .map_err(|err| format!("{}", err))?,
        ),
        (
            "pikevm_bytes_re",
            ExecBuilder::new(re)
                .nfa()
                .only_utf8(false)
                .build()
                .map(|exec| exec.into_byte_regex())
                .map_err(|err| format!("{}", err))?,
        ),
        (
            "default_bytes_re",
            ExecBuilder::new(re)
                .only_utf8(false)
                .build()
                .map(|exec| exec.into_byte_regex())
                .map_err(|err| format!("{}", err))?,
        ),
    ];

    Ok(string_checker::check_backends(&standard_backends)?
        + string_checker::check_backends(&utf8bytes_backends)?
        + bytes_checker::check_backends(&bytes_backends)?)
}

//
// A consistency checker parameterized by the input type (&str or &[u8]).
//

macro_rules! checker {
    ($module_name:ident, $regex_type:path, $mk_input:expr) => {
        mod $module_name {
            use quickcheck;
            use quickcheck::{Arbitrary, TestResult};

            pub fn check_backends(
                backends: &[(&str, $regex_type)],
            ) -> Result<u64, String> {
                let mut total_passed = 0;
                for regex in backends[1..].iter() {
                    total_passed += quickcheck_regex_eq(&backends[0], regex)?;
                }

                Ok(total_passed)
            }

            fn quickcheck_regex_eq(
                &(name1, ref re1): &(&str, $regex_type),
                &(name2, ref re2): &(&str, $regex_type),
            ) -> Result<u64, String> {
                quickcheck::QuickCheck::new()
                    .quicktest(RegexEqualityTest::new(
                        re1.clone(),
                        re2.clone(),
                    ))
                    .map_err(|err| {
                        format!(
                            "{}(/{}/) and {}(/{}/) are inconsistent.\
                             QuickCheck Err: {:?}",
                            name1, re1, name2, re2, err
                        )
                    })
            }

            struct RegexEqualityTest {
                re1: $regex_type,
                re2: $regex_type,
            }
            impl RegexEqualityTest {
                fn new(re1: $regex_type, re2: $regex_type) -> Self {
                    RegexEqualityTest { re1: re1, re2: re2 }
                }
            }

            impl quickcheck::Testable for RegexEqualityTest {
                fn result(&self, gen: &mut quickcheck::Gen) -> TestResult {
                    let input = $mk_input(gen);
                    let input = &input;

                    if self.re1.find(&input) != self.re2.find(input) {
                        return TestResult::error(format!(
                            "find mismatch input={:?}",
                            input
                        ));
                    }

                    let cap1 = self.re1.captures(input);
                    let cap2 = self.re2.captures(input);
                    match (cap1, cap2) {
                        (None, None) => {}
                        (Some(cap1), Some(cap2)) => {
                            for (c1, c2) in cap1.iter().zip(cap2.iter()) {
                                if c1 != c2 {
                                    return TestResult::error(format!(
                                        "captures mismatch input={:?}",
                                        input
                                    ));
                                }
                            }
                        }
                        _ => {
                            return TestResult::error(format!(
                                "captures mismatch input={:?}",
                                input
                            ))
                        }
                    }

                    let fi1 = self.re1.find_iter(input);
                    let fi2 = self.re2.find_iter(input);
                    for (m1, m2) in fi1.zip(fi2) {
                        if m1 != m2 {
                            return TestResult::error(format!(
                                "find_iter mismatch input={:?}",
                                input
                            ));
                        }
                    }

                    let ci1 = self.re1.captures_iter(input);
                    let ci2 = self.re2.captures_iter(input);
                    for (cap1, cap2) in ci1.zip(ci2) {
                        for (c1, c2) in cap1.iter().zip(cap2.iter()) {
                            if c1 != c2 {
                                return TestResult::error(format!(
                                    "captures_iter mismatch input={:?}",
                                    input
                                ));
                            }
                        }
                    }

                    let s1 = self.re1.split(input);
                    let s2 = self.re2.split(input);
                    for (chunk1, chunk2) in s1.zip(s2) {
                        if chunk1 != chunk2 {
                            return TestResult::error(format!(
                                "split mismatch input={:?}",
                                input
                            ));
                        }
                    }

                    TestResult::from_bool(true)
                }
            }
        } // mod
    }; // rule case
} // macro_rules!

checker!(string_checker, ::regex::Regex, |gen| String::arbitrary(gen));
checker!(bytes_checker, ::regex::bytes::Regex, |gen| Vec::<u8>::arbitrary(
    gen
));
