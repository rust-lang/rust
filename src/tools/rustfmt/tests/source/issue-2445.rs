test!(RunPassPretty {
            // comment
    path: "tests/run-pass/pretty",
    mode: "pretty",
    suite: "run-pass",
    default: false,
    host: true  // should, force, , no trailing comma here
});

test!(RunPassPretty {
            // comment
    path: "tests/run-pass/pretty",
    mode: "pretty",
    suite: "run-pass",
    default: false,
    host: true,         // should, , preserve, the trailing comma
});

test!(Test{
    field: i32, // comment
});
