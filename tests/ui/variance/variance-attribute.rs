#![feature(rustc_attrs)]
#![allow(internal_features, dead_code)]

#[rustc_dump_variances]
struct Cov<#[rustc_force_variance(covariant)] T: ?Sized>;
//~^ ERROR [T: +]
#[rustc_dump_variances]
struct Contra<#[rustc_force_variance(contravariant)] T: ?Sized>;
//~^ ERROR [T: -]
#[rustc_dump_variances]
struct Inv<#[rustc_force_variance(invariant)] T: ?Sized>;
//~^ ERROR [T: o]
#[rustc_dump_variances]
struct CovLifetime<#[rustc_force_variance(covariant)] 'a>;
//~^ ERROR ['a: +]
#[rustc_dump_variances]
struct ContraLifetime<#[rustc_force_variance(contravariant)] 'a>;
//~^ ERROR ['a: -]
#[rustc_dump_variances]
struct InvLifetime<#[rustc_force_variance(invariant)] 'a>;
//~^ ERROR ['a: o]

// Exercise propagation of explicit variance.
#[rustc_dump_variances]
struct Local<T: ?Sized, U: ?Sized, V: ?Sized>(Cov<T>, Contra<U>, Inv<V>);
//~^ ERROR [T: +, U: -, V: o]

// Override the invariance inferred from the field.
#[rustc_dump_variances]
struct Override<#[rustc_force_variance(covariant)] T: ?Sized>(Inv<T>);
//~^ ERROR [T: +]

// Only annotated parameters have their inferred variance overridden.
#[rustc_dump_variances]
struct Mixed<
    #[rustc_force_variance(contravariant)] 'a,
    T,
    #[rustc_force_variance(covariant)] U,
>(&'a T, Inv<U>);
//~^^^^^ ERROR ['a: -, T: +, U: +]

// Override a field's invariance with bivariance. The associated-type equality
// constrains T, so it is accepted without relaxing the unused-parameter check.
#[rustc_dump_variances]
struct Bivariant<#[rustc_force_variance(bivariant)] T, U: Iterator<Item = T>>(Inv<T>, U);
//~^ ERROR [T: *, U: +]

#[rustc_dump_variances]
struct BivariantLifetime<#[rustc_force_variance(bivariant)] 'a, U: Iterator<Item = &'a ()>>(
    InvLifetime<'a>,
    U,
);
//~^^^^ ERROR ['a: *, U: +]

fn main() {}
