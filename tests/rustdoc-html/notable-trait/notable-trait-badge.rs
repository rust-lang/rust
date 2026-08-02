#![feature(doc_notable_trait)]
#![crate_name = "foo"]

#[doc(notable_trait)]
pub trait Labeled {}

#[doc(notable_trait)]
pub trait AlsoLabeled {}

pub trait Plain {}

//@ has 'foo/struct.Tagged.html'
//@ has - '//div[@class="notable-trait-badge-container"]/a[@href="trait.Labeled.html"][@title="foo::Labeled"]' 'Labeled'
// Badges are sorted by trait name, so `AlsoLabeled` precedes `Labeled`.
//@ has - '//div[@class="notable-trait-badge-container"]/a[1]' 'ⓘ'
//@ has - '//div[@class="notable-trait-badge-container"]/a[2]' 'AlsoLabeled'
//@ has - '//div[@class="notable-trait-badge-container"]/a[3]' 'Labeled'
pub struct Tagged;
impl Labeled for Tagged {}
impl AlsoLabeled for Tagged {}
impl Plain for Tagged {}

//@ has 'foo/struct.Untagged.html'
//@ count - '//div[@class="notable-trait-badge-container"]' 0
pub struct Untagged;
impl Plain for Untagged {}

#[doc(notable_trait(color="grey"))]
pub trait Grey {}

#[doc(notable_trait(color="gray"))]
pub trait Gray {}

#[doc(notable_trait(color="blue"))]
pub trait Blue {}

//@ has 'foo/struct.Colors.html'
//@ count - '//a[@class="notable-trait-badge-grey"]' 2
//@ count - '//a[@class="notable-trait-badge-blue"]' 1
pub struct Colors;
impl Grey for Colors {}
impl Gray for Colors {}
impl Blue for Colors {}
