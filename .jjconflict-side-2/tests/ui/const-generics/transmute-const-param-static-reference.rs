//@ revisions: full adt_const_params min
//@[full] check-pass

#![cfg_attr(full, feature(adt_const_params, unsized_const_params))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(adt_const_params, feature(adt_const_params))]
#![cfg_attr(adt_const_params, allow(incomplete_features))]

struct Const<const P: &'static ()>;
//[min]~^ ERROR `&'static ()` is forbidden as the type of a const generic parameter
//[adt_const_params]~^^ ERROR use of unstable library feature `unsized_const_params`

fn main() {
    const A: &'static () = unsafe { std::mem::transmute(10 as *const ()) };

    let _ = Const::<{ A }>;
}
