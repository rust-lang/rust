#![crate_name = "foo"]

#[doc(hidden)]
pub trait Foo {}

trait Dark {}

pub trait Bam {}

pub struct Bar;

struct Hidden;

//@ !has foo/struct.Bar.html '//*[@id="impl-Foo-for-Bar"]' 'impl Foo for Bar'
impl Foo for Bar {}
//@ !has foo/struct.Bar.html '//*[@id="impl-Dark-for-Bar"]' 'impl Dark for Bar'
impl Dark for Bar {}
//@ has foo/struct.Bar.html '//*[@id="impl-Bam-for-Bar"]' 'impl Bam for Bar'
//@ has foo/trait.Bam.html '//*[@id="implementors-list"]' 'impl Bam for Bar'
impl Bam for Bar {}
//@ !has foo/trait.Bam.html '//*[@id="implementors-list"]' 'impl Bam for Hidden'
impl Bam for Hidden {}
