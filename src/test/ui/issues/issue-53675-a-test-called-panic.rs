// rust-lang/rust#53675: At one point the compiler errored when a test
// named `panic` used the `assert!` macro in expression position.

// build-pass (FIXME(62277): could be check-pass?)
// compile-flags: --test

mod in_expression_position {
    #[test]
    fn panic() {
        assert!(true)
    }
}

mod in_statement_position {
    #[test]
    fn panic() {
        assert!(true);
    }
}

mod what_if_we_use_panic_directly_in_expr {
    #[test]
    #[should_panic]
    fn panic() {
        panic!("in expr")
    }
}


mod what_if_we_use_panic_directly_in_stmt {
    #[test]
    #[should_panic]
    fn panic() {
        panic!("in stmt");
    }
}
