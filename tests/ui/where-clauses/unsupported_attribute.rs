#![crate_type = "lib"]
#![feature(alloc_error_handler)]
#![feature(cfg_accessible)]
#![feature(cfg_eval)]
#![feature(custom_test_frameworks)]
#![feature(derive_const)]
#![feature(where_clause_attrs)]
#![allow(soft_unstable)]

trait Trait {}

fn foo<'a, T>()
where
    #[doc = "doc"] T: Trait, //~ ERROR most attributes are not supported in `where` clauses
    #[doc = "doc"] 'a: 'static, //~ ERROR most attributes are not supported in `where` clauses
    #[ignore] T: Trait, //~ ERROR attribute cannot be used on
    #[ignore] 'a: 'static, //~ ERROR attribute cannot be used on
    #[should_panic] T: Trait, //~ ERROR attribute cannot be used on
    #[should_panic] 'a: 'static, //~ ERROR attribute cannot be used on
    #[macro_use] T: Trait, //~ ERROR attribute cannot be used on
    #[macro_use] 'a: 'static, //~ ERROR attribute cannot be used on
    #[allow(unused)] T: Trait, //~ ERROR most attributes are not supported in `where` clauses
    #[allow(unused)] 'a: 'static, //~ ERROR most attributes are not supported in `where` clauses
    #[deprecated] T: Trait, //~ ERROR attribute cannot be used on
    #[deprecated] 'a: 'static, //~ ERROR attribute cannot be used on
    #[automatically_derived] T: Trait, //~ ERROR attribute cannot be used on
    #[automatically_derived] 'a: 'static, //~ ERROR attribute cannot be used on
    #[derive(Clone)] T: Trait,
    //~^ ERROR most attributes are not supported in `where` clauses
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[derive(Clone)] 'a: 'static,
    //~^ ERROR most attributes are not supported in `where` clauses
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[rustfmt::skip] T: Trait, //~ ERROR most attributes are not supported in `where` clauses
    #[rustfmt::skip] 'a: 'static, //~ ERROR most attributes are not supported in `where` clauses
{}
