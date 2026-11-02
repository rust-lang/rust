//@aux-build:proc_macros.rs
//@no-rustfix
#![deny(clippy::deprecated_attributes_without_note)]

extern crate proc_macros;
use proc_macros::{external, with_span};

fn main() {
    // These should trigger the lint
    #[deprecated]
    //~^ deprecated_attributes_without_note
    fn no_fields() {}

    #[deprecated(since = "1.42.100")]
    //~^ deprecated_attributes_without_note
    fn has_since() {}

    #[cfg_attr(true, deprecated(since = "0.0.0"))]
    //~^ deprecated_attributes_without_note
    fn cfg_true_has_since() {}

    #[cfg_attr(true, deprecated)]
    //~^ deprecated_attributes_without_note
    fn cfg_true_no_fields() {}

    #[deprecated(since = "4.31.10")]
    //~^ deprecated_attributes_without_note
    mod module {
        #[deprecated]
        //~^ deprecated_attributes_without_note
        struct DeprecatedStruct {
            #[deprecated]
            //~^ deprecated_attributes_without_note
            deprecated_field: u32,
        }

        #[expect(deprecated, reason = "using a deprecated struct by way of impl for it")]
        #[deprecated]
        //~^ deprecated_attributes_without_note
        impl DeprecatedStruct {
            #[deprecated]
            //~^ deprecated_attributes_without_note
            fn deprecated_method() {}
        }

        #[deprecated]
        //~^ deprecated_attributes_without_note
        trait DeprecatedTrait {
            #[deprecated]
            //~^ deprecated_attributes_without_note
            fn trait_method();

            #[deprecated]
            //~^ deprecated_attributes_without_note
            fn default_trait_method() {}
        }
    }

    // These should be fine
    #[allow(deprecated)]
    #[allow(dead_code, reason = "This should be allowed")]
    #[expect(dead_code)]
    #[warn(dyn_drop, reason = "Warnings can also have reasons")]
    #[warn(redundant_lifetimes)]
    #[deny(deref_nullptr)]
    #[forbid(deref_nullptr)]
    fn correct_attribute_only() {}

    external! {
        #[deprecated]
        fn a() {}
    }
    with_span! {
        span
        #[deprecated]
        fn b() {}
    }

    #[deprecated(since = "TBD", note = "use has_both_note_first instead")]
    fn has_both_since_first() {}

    #[deprecated(note = "use has_since instead", since = "0.0.1")]
    fn has_both_note_first() {}

    #[deprecated(note = "I don't feel like maintaining this anymore, sorry")]
    fn has_note() {}

    #[deprecated = "probably a bad idea to use this"]
    fn has_note_eq() {}

    #[cfg_attr(false, deprecated(since = "255.255.255"))]
    fn cfg_false_has_since() {}

    #[cfg_attr(false, deprecated)]
    fn cfg_false_no_fields() {}

    #[cfg_attr(true, deprecated(note = "use cfg_false_has_both instead", since = "16.16.16"))]
    fn cfg_true_has_both() {}

    #[cfg_attr(false, deprecated(note = "use cfg_true_has_both instead", since = "19.19.19"))]
    fn cfg_false_has_both() {}

    #[cfg_attr(true, deprecated(note = "look down!"))]
    fn cfg_true_has_note() {}

    #[cfg_attr(false, deprecated(note = "look up!"))]
    fn cfg_false_has_note() {}

    #[cfg_attr(true, deprecated = "the note below me always lies")]
    fn cfg_true_has_note_eq() {}

    #[cfg_attr(false, deprecated = "the note above me never lies")]
    fn cfg_false_has_note_eq() {}

    #[deprecated = ""]
    fn empty_note() {}
}
