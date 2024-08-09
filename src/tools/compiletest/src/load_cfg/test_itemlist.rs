use super::*;

/// `(directive,  expected)`
const DIRECTIVE_CHECK: &[(&str, ItemVal<'static>)] = &[
    // set-once values
    ("build-pass", ItemVal::BuildPass),
    ("build-fail", ItemVal::BuildFail),
    ("check-pass", ItemVal::CheckPass),
    ("check-fail", ItemVal::CheckFail),
    ("run-pass", ItemVal::RunPass),
    ("run-fail", ItemVal::RunFail),
    ("no-prefer-dynamic", ItemVal::NoPreferDynamic),
    ("no-auto-check-cfg", ItemVal::NoAutoCheckCfg),
    // Boolean flags
    ("should-ice", ItemVal::ShouldIce(true)),
    ("no-should-ice", ItemVal::ShouldIce(false)),
    ("should-fail", ItemVal::ShouldFail(true)),
    ("no-should-fail", ItemVal::ShouldFail(false)),
    ("build-aux-docs", ItemVal::BuildAuxDocs(true)),
    ("no-build-aux-docs", ItemVal::BuildAuxDocs(false)),
    ("force-host", ItemVal::ForceHost(true)),
    ("no-force-host", ItemVal::ForceHost(false)),
    ("check-stdout", ItemVal::CheckStdout(true)),
    ("no-check-stdout", ItemVal::CheckStdout(false)),
    ("check-run-results", ItemVal::CheckRunResults(true)),
    ("no-check-run-results", ItemVal::CheckRunResults(false)),
    ("dont-check-compiler-stdout", ItemVal::DontCheckCompilerStdout(true)),
    ("no-dont-check-compiler-stdout", ItemVal::DontCheckCompilerStdout(false)),
    ("dont-check-compiler-stderr", ItemVal::DontCheckCompilerStderr(true)),
    ("no-dont-check-compiler-stderr", ItemVal::DontCheckCompilerStderr(false)),
    ("pretty-expanded", ItemVal::PrettyExpanded(true)),
    ("no-pretty-expanded", ItemVal::PrettyExpanded(false)),
    ("pretty-compare-only", ItemVal::PrettyCompareOnly(true)),
    ("no-pretty-compare-only", ItemVal::PrettyCompareOnly(false)),
    ("check-test-line-numbers-match", ItemVal::CheckTestLineNumbersMatch(true)),
    ("no-check-test-line-numbers-match", ItemVal::CheckTestLineNumbersMatch(false)),
    ("stderr-per-bitwidth", ItemVal::StderrPerBitwidth(true)),
    ("no-stderr-per-bitwidth", ItemVal::StderrPerBitwidth(false)),
    ("incremental", ItemVal::Incremental(true)),
    ("no-incremental", ItemVal::Incremental(false)),
    ("dont-check-failure-status", ItemVal::DontCheckFailureStatus(true)),
    ("no-dont-check-failure-status", ItemVal::DontCheckFailureStatus(false)),
    ("run-rustfix", ItemVal::RunRustfix(true)),
    ("no-run-rustfix", ItemVal::RunRustfix(false)),
    ("rustfix-only-machine-applicable", { ItemVal::RustfixOnlyMachineApplicable(true) }),
    ("no-rustfix-only-machine-applicable", { ItemVal::RustfixOnlyMachineApplicable(false) }),
    ("compare-output-lines-by-subset", ItemVal::CompareOutputLinesBySubset(true)),
    ("no-compare-output-lines-by-subset", ItemVal::CompareOutputLinesBySubset(false)),
    ("known-bug", ItemVal::KnownBug(true)),
    ("no-known-bug", ItemVal::KnownBug(false)),
    ("remap-src-base", ItemVal::RemapSrcBase(true)),
    ("no-remap-src-base", ItemVal::RemapSrcBase(false)),
    // Mappings
    ("error-pattern: testval", ItemVal::ErrorPattern("testval")),
    ("regex-error-pattern: testval", ItemVal::RegexErrorPattern("testval")),
    ("compile-flags: testval", ItemVal::CompileFlags("testval")),
    ("run-flags: testval", ItemVal::RunFlags("testval")),
    ("pretty-mode: testval", ItemVal::PrettyMode("testval")),
    ("aux-bin: testval", ItemVal::AuxBin("testval")),
    ("aux-build: testval", ItemVal::AuxBuild("testval")),
    ("aux-crate: testval", ItemVal::AuxCrate("testval")),
    ("aux-codegen-backend: testval", ItemVal::AuxCodegenBackend("testval")),
    ("exec-env: testval", ItemVal::ExecEnv("testval")),
    ("unset-exec-env: testval", ItemVal::UnsetExecEnv("testval")),
    ("rustc-env: testval", ItemVal::RustcEnv("testval")),
    ("unset-rustc-env: testval", ItemVal::UnsetRustcEnv("testval")),
    ("forbid-output: testval", ItemVal::ForbidOutput("testval")),
    ("failure-status: testval", ItemVal::FailureStatus("testval")),
    ("assembly-output: testval", ItemVal::AssemblyOutput("testval")),
    ("test-mir-pass: testval", ItemVal::TestMirPass("testval")),
    ("llvm-cov-flags: testval", ItemVal::LlvmCovFlags("testval")),
    ("filecheck-flags: testval", ItemVal::FilecheckFlags("testval")),
    ("revisions: testval", ItemVal::Revisions("testval")),
    // prefix-based rules
    ("ignore-foo", ItemVal::Ignore { what: "foo" }),
    ("needs-foo", ItemVal::Needs { what: "foo" }),
    ("only-foo", ItemVal::Only { what: "foo" }),
    ("normalize-foo", ItemVal::Normalize { what: "foo" }),
    // regex-based rules
    (
        "[rev1,rev-2] dir1 dir2",
        ItemVal::RevisionSpecificItems { revs: "rev1,rev-2", content: "dir1 dir2" },
    ),
    (
        "[1uNCOMmon_p4ttern-should_stillParse] anyth*ng & [e^erything!]",
        ItemVal::RevisionSpecificItems {
            revs: "1uNCOMmon_p4ttern-should_stillParse",
            content: "anyth*ng & [e^erything!]",
        },
    ),
];

const DIRECTIVE_ERRORS: &[&str] = &[""];

const GLOBAL_CHECK: &[(&str, ItemVal<'static>)] = &[
    ("//~ same line", ItemVal::UiDirective { revisions: None, adjust: None, content: "same line" }),
    (
        "unimporatant text //~ middle of line",
        ItemVal::UiDirective { revisions: None, adjust: None, content: "middle of line" },
    ),
    (
        "text//~ middle of line",
        ItemVal::UiDirective { revisions: None, adjust: None, content: "middle of line" },
    ),
    (
        "//~^ line up",
        ItemVal::UiDirective { revisions: None, adjust: Some("^"), content: "line up" },
    ),
    (
        "//~| adj 1 up",
        ItemVal::UiDirective { revisions: None, adjust: Some("|"), content: "adj 1 up" },
    ),
    (
        "//~^^^^ adj many up",
        ItemVal::UiDirective { revisions: None, adjust: Some("^^^^"), content: "adj many up" },
    ),
    (
        "//[abc,def-ghi]~ revisions same line",
        ItemVal::UiDirective {
            revisions: Some("abc,def-ghi"),
            adjust: None,
            content: "revisions same line",
        },
    ),
    (
        "//[abc]~^ revisions line up",
        ItemVal::UiDirective {
            revisions: Some("abc"),
            adjust: Some("^"),
            content: "revisions line up",
        },
    ),
    (
        "//[abc]~| revisions adj 1 up",
        ItemVal::UiDirective {
            revisions: Some("abc"),
            adjust: Some("|"),
            content: "revisions adj 1 up",
        },
    ),
    (
        "//[abc]~^^^^ revisions many up",
        ItemVal::UiDirective {
            revisions: Some("abc"),
            adjust: Some("^^^^"),
            content: "revisions many up",
        },
    ),
];

/// Things that should look like errors in global scope
const GLOBAL_ERRORS: &[&str] = &[
    // Regex patterns that are the wrong order. `(rev, sigil, adj)` is correct.
    "//~",             // standalone sigil
    "//[abc]",         // standalone rev
    "//^^",            // standalone adjustment
    "//[abc]^~",       // (rev, adj, sigil)
    "//~[abc,def]|",   // (sigil, rev, adj)
    "//~^^[abc def]",  // (sigil, adj, rev)
    "//^^ [abc def]~", // (adj, rev, sigil)
    "//|~[abc def]",   // (adj, sigil, rev)
    "// [abc] ^ ~",    // some things with whitespace
];

/// Simple unit test that directives get caught
#[test]
fn test_dir_match() {
    // (text, expected, actual)
    let mut errors = Vec::new();

    for (dir, expected) in DIRECTIVE_CHECK {
        let mut found = None;
        for matcher in &ITEM_MATCHERS {
            if let Some(v) = matcher.try_match_directive(dir, Pass::ExactOnly).expect("no errors") {
                found = Some(v);
            }
        }

        match found {
            Some(actual) if &actual == expected => (),
            Some(actual) => errors.push((dir, expected, Some(actual))),
            None => errors.push((dir, expected, None)),
        }
    }

    assert!(errors.is_empty(), "unmatched: {errors:#?}");
}

/// Add the comment and make sure we still match
#[test]
fn test_directives_as_global() {
    // (text, expected, actual)
    let mut errors = Vec::new();

    for (dir, expected) in DIRECTIVE_CHECK {
        for cty in CommentTy::all() {
            let line = format!("{} {dir}", cty.directive());
            let found = try_match_line(&line, *cty).expect("no errors");

            match found {
                Some(actual) if &actual == expected => (),
                Some(actual) => errors.push((line.clone(), expected, format!("{actual:?}"))),
                None => errors.push((line.clone(), expected, "None".to_owned())),
            }
        }
    }

    assert!(errors.is_empty(), "unmatched: {errors:#?}");
}

/// Try directives that are supposed to be global
#[test]
fn test_global() {
    // (text, expected, actual)
    let mut errors = Vec::new();

    for (line, expected) in GLOBAL_CHECK {
        for cty in CommentTy::all() {
            let line = line.replace("//", &cty.as_str());
            let found = try_match_line(&line, *cty).expect("no errors");

            match found {
                Some(actual) if &actual == expected => (),
                Some(actual) => errors.push((line.clone(), expected, format!("{actual:?}"))),
                None => errors.push((line.clone(), expected, "None".to_owned())),
            }
        }
    }

    assert!(errors.is_empty(), "unmatched: {errors:#?}");
}

/// Try directives that are supposed to be global
#[test]
fn test_global_errors() {
    // (text, expected, actual)
    let mut unexpected_ok = Vec::new();

    for line in GLOBAL_ERRORS {
        for cty in CommentTy::all() {
            let line = line.replace("//", &cty.as_str());
            let res = try_match_line(&line, *cty);

            if res.is_ok() {
                // We should only see `Err` values.
                unexpected_ok.push((line.clone(), format!("{res:?}")));
            }
        }
    }

    assert!(unexpected_ok.is_empty(), "should parse as errors but didn't: {unexpected_ok:#?}");
}

const SAMPLE_FILE: &str = r#"
//@ ignore-armv7
//@ compile-flags: -O -C no-prepopulate-passes
//@ revisions: x86 other
//@[x86] should-fail
//@[other] compile-flags: -O

// CHECK: define abcd
fn foo() {}

#![feature(...)]
//~^ WARN be careful
//~| WARN seriously!

fn fun()
//[rev-1]~^ ERROR no fun allowed
{}
"#;

const SAMPLE_EXPECTED: LazyCell<Vec<Item>> = LazyCell::new(|| {
    vec![
        ItemVal::Ignore { what: "armv7" }.to_item(2, 1),
        ItemVal::CompileFlags("-O -C no-prepopulate-passes").to_item(3, 1),
        ItemVal::Revisions("x86 other").to_item(4, 1),
        ItemVal::RevisionSpecificExpanded {
            revs: "x86",
            content: Box::new(ItemVal::ShouldFail(true)),
        }
        .to_item(5, 1),
        ItemVal::RevisionSpecificExpanded {
            revs: "other",
            content: Box::new(ItemVal::CompileFlags("-O")),
        }
        .to_item(6, 1),
        ItemVal::FileCheckDirective { directive: "CHECK", content: Some("define abcd") }
            .to_item(8, 1),
        ItemVal::UiDirective { revisions: None, adjust: Some("^"), content: "WARN be careful" }
            .to_item(12, 1),
        ItemVal::UiDirective { revisions: None, adjust: Some("|"), content: "WARN seriously!" }
            .to_item(13, 1),
        ItemVal::UiDirective {
            revisions: Some("rev-1"),
            adjust: Some("^"),
            content: "ERROR no fun allowed",
        }
        .to_item(16, 1),
    ]
});

#[test]
fn test_whole_list() {
    let parsed = parse(SAMPLE_FILE, CommentTy::Slashes).unwrap();
    assert_eq!(parsed, *SAMPLE_EXPECTED, "{parsed:#?}\n{:#?}", *SAMPLE_EXPECTED);
}
