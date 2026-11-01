// Given an enum `Enum` with lifetime parameters, we used to lower `Enum::<…>::Variant {}` to
// `Enum::<…>::Variant::<'_, …> {}`, i.e., synthesize `'_` lifetime args for enum variants
// even if generic args were already passed to the enum itself.
// Obviously, these conflicting generic arg lists were then rejected later on (by HIR ty lowering).
//
// Now we no longer do. Why do we synthesize these lifetimes for struct variant paths in the first
// place while we don't do it for unit & tuple variants? Well, we later HIR-ty-lower the former in
// "type mode" while we lower the latter in "value mode". In "type mode", we won't infer
// elided lifetimes if the generic arg lists contains non-lifetime args which is undesirable here.
//
// issue: <https://github.com/rust-lang/rust/issues/108224>
//@ check-pass

fn scope<'any>() {
    enum Ty0<'a> {
        Unit,
        Tuple,
        Struct {},

        Carry(&'a ()),
    }

    enum Ty1<'a, T: ?Sized> {
        Variant,

        Carry(&'a T),
    }

    let _ = Ty0::<'_>::Unit {};
    let _ = Ty0::<'static>::Tuple {};
    let _ = Ty0::<'any>::Struct {};

    let _ = Ty1::</*'_, */()>::Variant {};
    let _ = Ty1::<'any, ()>::Variant {};
}

fn main() {}
