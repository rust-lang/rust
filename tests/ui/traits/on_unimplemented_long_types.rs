//@ compile-flags: --diagnostic-width=60 -Z write-long-types-to-disk=yes
//@ normalize-stderr-test: "long-type-\d+" -> "long-type-hash"

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
