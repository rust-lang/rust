#[doc(alias="DocAliasS1")]
pub struct S1;

#[doc(alias="DocAliasS2")]
#[doc(alias("DocAliasS3", "DocAliasS4"))]
pub struct S2;

#[doc(alias("doc_alias_f1", "doc_alias_f2"))]
pub fn f() {}

pub mod m {
    #[doc(alias="DocAliasS5")]
    pub struct S5;
}
