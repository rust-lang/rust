#![crate_type = "lib"]
#![feature(alloc_error_handler)]
#![feature(cfg_accessible)]
#![feature(cfg_eval)]
#![feature(custom_test_frameworks)]
#![feature(derive_const)]
#![feature(where_clause_attrs)]
#![feature(stmt_expr_attributes)]
#![allow(soft_unstable)]

trait Trait {}

fn foo<'a, T>()
where
    #[doc = "doc"] T: Trait, //~ ERROR most attributes are not supported in `where` clauses
    #[doc = "doc"] 'a: 'static, //~ ERROR most attributes are not supported in `where` clauses
    #[ignore] T: Trait, //~ ERROR most attributes are not supported in `where` clauses
    #[ignore] 'a: 'static, //~ ERROR most attributes are not supported in `where` clauses
    #[should_panic] T: Trait, //~ ERROR most attributes are not supported in `where` clauses
    #[should_panic] 'a: 'static, //~ ERROR most attributes are not supported in `where` clauses
    #[macro_use] T: Trait, //~ ERROR most attributes are not supported in `where` clauses
    #[macro_use] 'a: 'static, //~ ERROR most attributes are not supported in `where` clauses
    #[allow(unused)] T: Trait, //~ ERROR most attributes are not supported in `where` clauses
    #[allow(unused)] 'a: 'static, //~ ERROR most attributes are not supported in `where` clauses
    #[deprecated] T: Trait, //~ ERROR most attributes are not supported in `where` clauses
    #[deprecated] 'a: 'static, //~ ERROR most attributes are not supported in `where` clauses
    #[automatically_derived] T: Trait, //~ ERROR most attributes are not supported in `where` clauses
    #[automatically_derived] 'a: 'static, //~ ERROR most attributes are not supported in `where` clauses
    #[derive(Clone)] T: Trait,
    //~^ ERROR most attributes are not supported in `where` clauses
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[derive(Clone)] 'a: 'static,
    //~^ ERROR most attributes are not supported in `where` clauses
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[rustfmt::skip] T: Trait, //~ ERROR most attributes are not supported in `where` clauses
    #[rustfmt::skip] 'a: 'static, //~ ERROR most attributes are not supported in `where` clauses
    #[must_use] T: Trait, //~ ERROR most attributes are not supported in `where` clauses
    #[must_use] 'a: 'static, //~ ERROR most attributes are not supported in `where` clauses
    #[cold] T: Trait, //~ ERROR most attributes are not supported in `where` clauses
    #[cold] 'a: 'static, //~ ERROR most attributes are not supported in `where` clauses
    #[repr()] T: Trait, //~ ERROR most attributes are not supported in `where` clauses
    #[repr()] 'a: 'static, //~ ERROR most attributes are not supported in `where` clauses
{}

fn another_one() {
    // Regression test for https://github.com/rust-lang/rust/issues/143787
    let _: String = #[repr()] std::string::String::new();
}
