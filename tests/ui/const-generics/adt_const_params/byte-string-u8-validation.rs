// Check that a byte string literal to a const parameter with a non-u8
// element type isn't lowered to a ValTree with an incorrect type

#![feature(adt_const_params)]
#![feature(rustc_attrs)]

#[rustc_dump_predicates]
struct ConstBytes<const T: &'static [*mut u8; 3]>
//~^ ERROR rustc_dump_predicates
//~| NOTE Binder { value: ConstArgHasType(T/#0, &'static [*mut u8; 3_usize]), bound_vars: [] }
//~| NOTE Binder { value: TraitPredicate(<ConstBytes<{const error}> as std::marker::Sized>, polarity:Positive), bound_vars: [] }
where
    ConstBytes<b"AAA">: Sized;
//~^ ERROR mismatched types
//~| NOTE expected `&[*mut u8; 3]`, found `&[u8; 3]`
//~| NOTE expected reference `&'static [*mut u8; 3]`

fn main() {}
