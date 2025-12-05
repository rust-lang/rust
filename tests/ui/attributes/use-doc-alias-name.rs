//@ aux-build: use-doc-alias-name-extern.rs

// issue#124273

extern crate use_doc_alias_name_extern;

use use_doc_alias_name_extern::*;

#[doc(alias="LocalDocAliasS")]
struct S;

fn main() {
    LocalDocAliasS; // don't show help in local crate
    //~^ ERROR: cannot find value `LocalDocAliasS` in this scope

    DocAliasS1;
    //~^ ERROR: cannot find value `DocAliasS1` in this scope
    //~| HELP: `S1` has a name defined in the doc alias attribute as `DocAliasS1`

    DocAliasS2;
    //~^ ERROR: cannot find value `DocAliasS2` in this scope
    //~| HELP: `S2` has a name defined in the doc alias attribute as `DocAliasS2`

    DocAliasS3;
    //~^ ERROR: cannot find value `DocAliasS3` in this scope
    //~| HELP: `S2` has a name defined in the doc alias attribute as `DocAliasS3`

    DocAliasS4;
    //~^ ERROR: cannot find value `DocAliasS4` in this scope
    //~| HELP: `S2` has a name defined in the doc alias attribute as `DocAliasS4`

    doc_alias_f1();
    //~^ ERROR: cannot find function `doc_alias_f1` in this scope
    //~| HELP: `f` has a name defined in the doc alias attribute as `doc_alias_f1`

    doc_alias_f2();
    //~^ ERROR: cannot find function `doc_alias_f2` in this scope
    //~| HELP: `f` has a name defined in the doc alias attribute as `doc_alias_f2`

    m::DocAliasS5;
    //~^ ERROR: cannot find value `DocAliasS5` in module `m`
    //~| HELP: `S5` has a name defined in the doc alias attribute as `DocAliasS5`

    not_exist_module::DocAliasS1;
    //~^ ERROR: use of unresolved module or unlinked crate `not_exist_module`
    //~| HELP: you might be missing a crate named `not_exist_module`

    use_doc_alias_name_extern::DocAliasS1;
    //~^ ERROR: cannot find value `DocAliasS1` in crate `use_doc_alias_name_extern
    //~| HELP: `S1` has a name defined in the doc alias attribute as `DocAliasS1`

    m::n::DocAliasX::y::S6;
    //~^ ERROR: could not find `DocAliasX` in `n`
    //~| HELP: `x` has a name defined in the doc alias attribute as `DocAliasX`

    m::n::x::y::DocAliasS6;
    //~^ ERROR: cannot find value `DocAliasS6` in module `m::n::x::y`
    //~| HELP: `S6` has a name defined in the doc alias attribute as `DocAliasS6`
}

trait T {
    fn f() {
        DocAliasS1;
        //~^ ERROR: cannot find value `DocAliasS1` in this scope
        //~| HELP: `S1` has a name defined in the doc alias attribute as `DocAliasS1`
    }
}
