/*
 * This test is a minimal version of <rofl_0> and <subdiff_0>
 *
 * Once this bug gets fixed, uncomment rofl_0 and subdiff_0
 * (in `tests/crates_regex.rs`).
#[test]
fn word_boundary_backtracking_default_mismatch() {
    use regex::internal::ExecBuilder;

    let backtrack_re = ExecBuilder::new(r"\b")
        .bounded_backtracking()
        .build()
        .map(|exec| exec.into_regex())
        .map_err(|err| format!("{}", err))
        .unwrap();

    let default_re = ExecBuilder::new(r"\b")
        .build()
        .map(|exec| exec.into_regex())
        .map_err(|err| format!("{}", err))
        .unwrap();

    let input = "䅅\\u{a0}";

    let fi1 = backtrack_re.find_iter(input);
    let fi2 = default_re.find_iter(input);
    for (m1, m2) in fi1.zip(fi2) {
        assert_eq!(m1, m2);
    }
}
*/

mod consistent;

mod crates_regex {

    macro_rules! consistent {
        ($test_name:ident, $regex_src:expr) => {
            #[test]
            fn $test_name() {
                use super::consistent::backends_are_consistent;

                if option_env!("RUST_REGEX_RANDOM_TEST").is_some() {
                    match backends_are_consistent($regex_src) {
                        Ok(_) => {}
                        Err(err) => panic!("{}", err),
                    }
                }
            }
        };
    }

    include!("crates_regex.rs");
}
