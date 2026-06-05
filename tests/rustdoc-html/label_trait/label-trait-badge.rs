#![feature(doc_label_trait)]
#![crate_name = "foo"]

#[doc(label_trait)]
pub trait Labeled {}

#[doc(label_trait)]
pub trait AlsoLabeled {}

pub trait Plain {}

//@ has 'foo/struct.Tagged.html'
//@ has - '//a[@class="impl-label-trait-full-badge"][@href="trait.Labeled.html"][@title="foo::Labeled"]' 'Labeled'
// Badges are sorted by trait name, so `AlsoLabeled` precedes `Labeled`.
//@ has - '//div[@class="impl-label-trait-full-badge-container"]/a[1]' 'AlsoLabeled'
//@ has - '//div[@class="impl-label-trait-full-badge-container"]/a[2]' 'Labeled'
pub struct Tagged;
impl Labeled for Tagged {}
impl AlsoLabeled for Tagged {}
impl Plain for Tagged {}

//@ has 'foo/struct.Untagged.html'
//@ count - '//div[@class="impl-label-trait-full-badge-container"]' 0
pub struct Untagged;
impl Plain for Untagged {}
