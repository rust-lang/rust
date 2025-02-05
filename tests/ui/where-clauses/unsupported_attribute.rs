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
    #[doc = "doc"] T: Trait, //~ ERROR most attributes in `where` clauses are not supported
    #[doc = "doc"] 'a: 'static, //~ ERROR most attributes in `where` clauses are not supported
    #[ignore] T: Trait, //~ ERROR most attributes in `where` clauses are not supported
    #[ignore] 'a: 'static, //~ ERROR most attributes in `where` clauses are not supported
    #[should_panic] T: Trait, //~ ERROR most attributes in `where` clauses are not supported
    #[should_panic] 'a: 'static, //~ ERROR most attributes in `where` clauses are not supported
    #[macro_use] T: Trait, //~ ERROR most attributes in `where` clauses are not supported
    #[macro_use] 'a: 'static, //~ ERROR most attributes in `where` clauses are not supported
    #[allow(unused)] T: Trait, //~ ERROR most attributes in `where` clauses are not supported
    #[allow(unused)] 'a: 'static, //~ ERROR most attributes in `where` clauses are not supported
    #[warn(unused)] T: Trait, //~ ERROR most attributes in `where` clauses are not supported
    #[warn(unused)] 'a: 'static, //~ ERROR most attributes in `where` clauses are not supported
    #[deny(unused)] T: Trait, //~ ERROR most attributes in `where` clauses are not supported
    #[deny(unused)] 'a: 'static, //~ ERROR most attributes in `where` clauses are not supported
    #[forbid(unused)] T: Trait, //~ ERROR most attributes in `where` clauses are not supported
    #[forbid(unused)] 'a: 'static, //~ ERROR most attributes in `where` clauses are not supported
    #[deprecated] T: Trait, //~ ERROR most attributes in `where` clauses are not supported
    #[deprecated] 'a: 'static, //~ ERROR most attributes in `where` clauses are not supported
    #[automatically_derived] T: Trait, //~ ERROR most attributes in `where` clauses are not supported
    #[automatically_derived] 'a: 'static, //~ ERROR most attributes in `where` clauses are not supported
    #[derive(Clone)] T: Trait,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[derive(Clone)] 'a: 'static,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `derive`
    #[global_allocator] T: Trait,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `global_allocator`
    #[global_allocator] 'a: 'static,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `global_allocator`
    #[test] T: Trait,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `test`
    #[test] 'a: 'static,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `test`
    #[alloc_error_handler] T: Trait,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `alloc_error_handler`
    #[alloc_error_handler] 'a: 'static,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `alloc_error_handler`
    #[bench] T: Trait,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `bench`
    #[bench] 'a: 'static,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `bench`
    #[cfg_accessible] T: Trait,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `cfg_accessible`
    #[cfg_accessible] 'a: 'static,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `cfg_accessible`
    #[cfg_eval] T: Trait,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `cfg_eval`
    #[cfg_eval] 'a: 'static,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `cfg_eval`
    #[derive_const(Clone)] T: Trait,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `derive_const`
    #[derive_const(Clone)] 'a: 'static,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `derive_const`
    #[test_case] T: Trait,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `test_case`
    #[test_case] 'a: 'static,
    //~^ ERROR most attributes in `where` clauses are not supported
    //~| ERROR expected non-macro attribute, found attribute macro `test_case`
    #[rustfmt::skip] T: Trait, //~ ERROR most attributes in `where` clauses are not supported
    #[rustfmt::skip] 'a: 'static, //~ ERROR most attributes in `where` clauses are not supported
{}
