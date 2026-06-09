// Checks that you can set a lint level specficially for a macro definition.
//
// This is a regression test for issue #59306.
//
//@ check-pass


#[deny(missing_docs)]
mod module {
    #[allow(missing_docs)]
    #[macro_export]
    macro_rules! hello {
        () => ()
    }
}

fn main() {}
