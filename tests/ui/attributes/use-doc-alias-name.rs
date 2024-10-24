//@ aux-build: use-doc-alias-name-extern.rs
//@ error-pattern: `S1` has a name defined in the doc alias attribute as `DocAliasS1`
//@ error-pattern: `S2` has a name defined in the doc alias attribute as `DocAliasS2`
//@ error-pattern: `S2` has a name defined in the doc alias attribute as `DocAliasS3`
//@ error-pattern: `S2` has a name defined in the doc alias attribute as `DocAliasS4`
//@ error-pattern: `f` has a name defined in the doc alias attribute as `doc_alias_f1`
//@ error-pattern: `f` has a name defined in the doc alias attribute as `doc_alias_f2`
//@ error-pattern: `S5` has a name defined in the doc alias attribute as `DocAliasS5`

// issue#124273

extern crate use_doc_alias_name_extern;

use use_doc_alias_name_extern::*;

#[doc(alias="LocalDocAliasS")]
struct S;

fn main() {
    LocalDocAliasS;
    //~^ ERROR: cannot find value `LocalDocAliasS` in this scope
    DocAliasS1;
    //~^ ERROR: cannot find value `DocAliasS1` in this scope
    DocAliasS2;
    //~^ ERROR: cannot find value `DocAliasS2` in this scope
    DocAliasS3;
    //~^ ERROR: cannot find value `DocAliasS3` in this scope
    DocAliasS4;
    //~^ ERROR: cannot find value `DocAliasS4` in this scope
    doc_alias_f1();
    //~^ ERROR: cannot find function `doc_alias_f1` in this scope
    doc_alias_f2();
    //~^ ERROR: cannot find function `doc_alias_f2` in this scope
    m::DocAliasS5;
    //~^ ERROR: cannot find value `DocAliasS5` in module `m`
}
