// For historical reasons, item modifier `default` doesn't have a proper pre-expansion feature gate.
// We're now at least issuing a *warning* for those that only exist before macro expansion.
// FIXME(#154045): Turn their post-expansion feature gate into a proper pre-expansion one.
//                 As part of this, move these test cases into `feature-gate-specialization.rs`.
//
// Moreover, `specialization` implies `min_specialization` similar to the post-expansion gate.
//
// However, while we only gate `default` *associated* functions only behind `min_specialization` OR
// `specialization` in the post-expansion case, in the pre-expansion case we gate all kinds of
// functions (free, assoc, foreign) behind `min_specialization` OR `specialization` if marked with
// `default` for simplicity of implementation. Ultimately it doesn't matter since we later reject
// `default` on anything other than impls & impl assoc items during semantic analysis.
//
//@ revisions: default min full
//@ check-pass
#![cfg_attr(min, feature(min_specialization))]
#![cfg_attr(full, feature(specialization))]

#[cfg(false)]
impl Trait for () {
    default type Ty = ();
    //[default,min]~^ WARN specialization is experimental
    //[default,min]~| WARN unstable syntax can change at any point in the future
    default const CT: () = ();
    //[default,min]~^ WARN specialization is experimental
    //[default,min]~| WARN unstable syntax can change at any point in the future
    default fn fn_();
    //[default]~^ WARN specialization is experimental
    //[default]~| WARN unstable syntax can change at any point in the future
}

// While free ty/ct/fn items marked `default` are
// semantically malformed we still need to gate the keyword!
#[cfg(false)]
default fn fn_() {}
//[default]~^ WARN specialization is experimental
//[default]~| WARN unstable syntax can change at any point in the future

#[cfg(false)]
default impl Trait for () {}
//[default,min]~^ WARN specialization is experimental
//[default,min]~| WARN unstable syntax can change at any point in the future

fn main() {}
