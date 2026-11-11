//@ check-pass
//@ compile-flags:--test
//@ normalize-stdout: "tests/rustdoc-ui/issues" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"

pub fn test() -> Result<(), ()> {
    //! ```compile_fail
    //! fn test() -> Result< {}
    //! ```
    Ok(())
}
