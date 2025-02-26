//@ compile-flags: --diagnostic-width=60 -Z write-long-types-to-disk=yes
// The regex below normalizes the long type file name to make it suitable for compare-modes.
//@ normalize-stderr: "'\$TEST_BUILD_DIR/.*\.long-type-\d+.txt'" -> "'$$TEST_BUILD_DIR/$$FILE.long-type-hash.txt'"

pub fn foo() -> impl std::fmt::Display {
    //~^ ERROR doesn't implement `std::fmt::Display`
    Some(Some(Some(Some(Some(Some(Some(Some(Some(Some(Some(
        Some(Some(Some(Some(Some(Some(Some(Some(Some(Some(Some(
            Some(Some(Some(Some(Some(Some(Some(Some(Some(Some(Some(
                Some(Some(Some(Some(Some(Some(Some(Some(Some(Some(Some(
                    Some(Some(Some(Some(Some(Some(Some(Some(())))))))),
                ))))))))))),
            ))))))))))),
        ))))))))))),
    )))))))))))
}

fn main() {}
