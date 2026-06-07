// The presence of built-in or user-written trait impls disqualifies potential built-in auto trait
// impls for a type. Here we have a blanket impl of an auto trait which suppresses all hypothetical
// auto trait impls.
//
// Check that we don't claim that there's an auto trait impl & ensure that we show the blanket impl.
// Erroneously we once used to omit the blanket impl since auto trait impls & blanket impls used to
// share the same cache and auto trait impls get generated first. Now, they have separate caches.
//
// issue: <https://github.com/rust-lang/rust/issues/148980>

#![feature(auto_traits)]
#![crate_name = "it"]

pub auto trait Marker {}

//@ has 'it/struct.Subject.html'
#[derive(Clone, Copy)]
pub struct Subject;

// NOTE: `#synthetic-implementations-list` contains auto trait impls only despite its name.
//@ !has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]' \
//        'Marker for Subject'

//@ has - '//*[@id="blanket-implementations-list"]//*[@class="impl"]' \
//        'Marker for Twhere T: Copy'
impl<T: Copy> Marker for T {}
