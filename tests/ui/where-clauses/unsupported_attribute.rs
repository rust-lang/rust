#![crate_type = "lib"]
#![feature(cfg_attribute_in_where)]

trait Trait {}

fn foo<'a, T>()
where
    #[doc = "doc"] T: Trait, //~ ERROR attributes in `where` clauses are not supported
    #[doc = "doc"] 'a: 'static, //~ ERROR attributes in `where` clauses are not supported
    #[ignore] T: Trait, //~ ERROR attributes in `where` clauses are not supported
    #[ignore] 'a: 'static, //~ ERROR attributes in `where` clauses are not supported
    #[should_panic] T: Trait, //~ ERROR attributes in `where` clauses are not supported
    #[should_panic] 'a: 'static, //~ ERROR attributes in `where` clauses are not supported
    #[macro_use] T: Trait, //~ ERROR attributes in `where` clauses are not supported
    #[macro_use] 'a: 'static, //~ ERROR attributes in `where` clauses are not supported
    #[allow(unused)] T: Trait, //~ ERROR attributes in `where` clauses are not supported
    #[allow(unused)] 'a: 'static, //~ ERROR attributes in `where` clauses are not supported
    #[warn(unused)] T: Trait, //~ ERROR attributes in `where` clauses are not supported
    #[warn(unused)] 'a: 'static, //~ ERROR attributes in `where` clauses are not supported
    #[deny(unused)] T: Trait, //~ ERROR attributes in `where` clauses are not supported
    #[deny(unused)] 'a: 'static, //~ ERROR attributes in `where` clauses are not supported
    #[forbid(unused)] T: Trait, //~ ERROR attributes in `where` clauses are not supported
    #[forbid(unused)] 'a: 'static, //~ ERROR attributes in `where` clauses are not supported
    #[deprecated] T: Trait, //~ ERROR attributes in `where` clauses are not supported
    #[deprecated] 'a: 'static, //~ ERROR attributes in `where` clauses are not supported
    #[automatically_derived] T: Trait, //~ ERROR attributes in `where` clauses are not supported
    #[automatically_derived] 'a: 'static, //~ ERROR attributes in `where` clauses are not supported
{}
