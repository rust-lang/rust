//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver=globally

#![feature(
    gca_adts,
    gca_macroless_args,
    gca_macroless_items,
    gca_min_const_items,
    min_adt_const_params
)]

struct S<const N: (u8,)>;

struct Field {
    value: S<{ (1_u8,) }>,
}

trait Tr {
    fn f(_: S<{ (1_u8,) }>) {}
}

fn with_bound()
where
    S<{ (1_u8,) }>: Sized,
{
}

const VALUE: (u8,) = (1_u8,);
const FROM_FN: u8 = std::gca!(const { value() });

const fn value() -> u8 {
    VALUE.0
}

impl S<{ (1_u8,) }> {
    const VALUE: (u8,) = (1_u8,);
}

trait Constants {
    #[rustc_always_gca]
    const VALUE: (u8,);
}

impl Constants for () {
    const VALUE: (u8,) = (1_u8,);
}

fn generic<T: Constants>() {
    let _ = T::VALUE;
}

fn main() {
    let _ = VALUE;
    let _ = FROM_FN;
    let _ = S::<{ (1_u8,) }>::VALUE;
    let _ = <() as Constants>::VALUE;
    generic::<()>();
}
