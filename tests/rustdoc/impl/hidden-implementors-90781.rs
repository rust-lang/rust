//@ compile-flags: -Z unstable-options --document-hidden-items --document-private-items

// regression test for https://github.com/rust-lang/rust/issues/90781
#![crate_name = "foo"]

//@ has foo/trait.TPubVis.html
//@ has - '//*[@id="implementors-list"]' 'HidPriv'
//@ has - '//*[@id="implementors-list"]' 'HidPub'
//@ has - '//*[@id="implementors-list"]' 'VisPriv'
//@ has - '//*[@id="implementors-list"]' 'VisPub'
pub trait TPubVis {}

//@ has foo/trait.TPubHidden.html
//@ has - '//*[@id="implementors-list"]' 'HidPriv'
//@ has - '//*[@id="implementors-list"]' 'HidPub'
//@ has - '//*[@id="implementors-list"]' 'VisPriv'
//@ has - '//*[@id="implementors-list"]' 'VisPub'
#[doc(hidden)]
pub trait TPubHidden {}

//@ has foo/trait.TPrivVis.html
//@ has - '//*[@id="implementors-list"]' 'HidPriv'
//@ has - '//*[@id="implementors-list"]' 'HidPub'
//@ has - '//*[@id="implementors-list"]' 'VisPriv'
//@ has - '//*[@id="implementors-list"]' 'VisPub'
trait TPrivVis {}

#[doc(hidden)]
//@ has foo/trait.TPrivHidden.html
//@ has - '//*[@id="impl-TPrivHidden-for-HidPriv"]' 'HidPriv'
//@ has - '//*[@id="impl-TPrivHidden-for-HidPub"]' 'HidPub'
//@ has - '//*[@id="impl-TPrivHidden-for-VisPriv"]' 'VisPriv'
//@ has - '//*[@id="impl-TPrivHidden-for-VisPub"]' 'VisPub'
trait TPrivHidden {}

//@ has foo/struct.VisPub.html
//@ has - '//*[@id="trait-implementations-list"]' 'TPrivHidden'
//@ has - '//*[@id="trait-implementations-list"]' 'TPrivVis'
//@ has - '//*[@id="trait-implementations-list"]' 'TPubHidden'
//@ has - '//*[@id="trait-implementations-list"]' 'TPubVis'
pub struct VisPub;

//@ has foo/struct.VisPriv.html
//@ has - '//*[@id="trait-implementations-list"]' 'TPrivHidden'
//@ has - '//*[@id="trait-implementations-list"]' 'TPrivVis'
//@ has - '//*[@id="trait-implementations-list"]' 'TPubHidden'
//@ has - '//*[@id="trait-implementations-list"]' 'TPubVis'
struct VisPriv;

//@ has foo/struct.HidPub.html
//@ has - '//*[@id="trait-implementations-list"]' 'TPrivHidden'
//@ has - '//*[@id="trait-implementations-list"]' 'TPrivVis'
//@ has - '//*[@id="trait-implementations-list"]' 'TPubHidden'
//@ has - '//*[@id="trait-implementations-list"]' 'TPubVis'
#[doc(hidden)]
pub struct HidPub;

//@ has foo/struct.HidPriv.html
//@ has - '//*[@id="trait-implementations-list"]' 'TPrivHidden'
//@ has - '//*[@id="trait-implementations-list"]' 'TPrivVis'
//@ has - '//*[@id="trait-implementations-list"]' 'TPubHidden'
//@ has - '//*[@id="trait-implementations-list"]' 'TPubVis'
#[doc(hidden)]
struct HidPriv;

macro_rules! implement {
    ($trait:ident - $($struct:ident)+) => {
        $(
            impl $trait for $struct {}
        )+
    }
}


implement!(TPubVis - VisPub VisPriv HidPub HidPriv);
implement!(TPubHidden - VisPub VisPriv HidPub HidPriv);
implement!(TPrivVis - VisPub VisPriv HidPub HidPriv);
implement!(TPrivHidden - VisPub VisPriv HidPub HidPriv);
