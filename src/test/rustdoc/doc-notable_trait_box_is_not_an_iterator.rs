#![feature(doc_notable_trait)]
#![feature(lang_items)]
#![feature(no_core)]
#![no_core]
#[lang = "owned_box"]
pub struct Box<T>;

impl<T> Box<T> {
    pub fn new(x: T) -> Box<T> {
        Box
    }
}

#[doc(notable_trait)]
pub trait FakeIterator {}

impl<I: FakeIterator> FakeIterator for Box<I> {}

#[lang = "pin"]
pub struct Pin<T>;

impl<T> Pin<T> {
    pub fn new(x: T) -> Pin<T> {
        Pin
    }
}

impl<I: FakeIterator> FakeIterator for Pin<I> {}

// @!has doc_notable_trait_box_is_not_an_iterator/fn.foo.html '//*' 'Notable'
pub fn foo<T>(x: T) -> Box<T> {
    Box::new(x)
}

// @!has doc_notable_trait_box_is_not_an_iterator/fn.bar.html '//*' 'Notable'
pub fn bar<T>(x: T) -> Pin<T> {
    Pin::new(x)
}
