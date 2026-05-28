#![crate_type = "lib"]

// This is part of a collection of regression tests related to the NLL problem case 3 that was
// deferred from the implementation of the NLL RFC, and left to be implemented by polonius. They are
// from open issues, e.g. tagged fixed-by-polonius, to ensure that the polonius alpha analysis does
// handle them, as does the datalog implementation.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius legacy
//@ [nll] known-bug: #123839
//@ [polonius] check-pass
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] check-pass
//@ [legacy] compile-flags: -Z polonius=legacy

struct Foo {
    val: i32,
    status: i32,
    err_str: String,
}

impl Foo {
    fn bar(self: &mut Self) -> Result<(), &str> {
        if self.val == 0 {
            self.status = -1;
            Err("val is zero")
        } else if self.val < 0 {
            self.status = -2;
            self.err_str = format!("unexpected negative val {}", self.val);
            Err(&self.err_str)
        } else {
            Ok(())
        }
    }
    fn foo(self: &mut Self) -> Result<(), &str> {
        self.bar()?; // rust reports this line conflicts with the next line
        self.status = 1; // and this line is the victim
        Ok(())
    }
}
