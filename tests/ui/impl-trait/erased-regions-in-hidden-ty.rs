// revisions: current next
// compile-flags: -Zverbose
//[next] compile-flags: -Ztrait-solver=next
// normalize-stderr-test "DefId\([^\)]+\)" -> "DefId(..)"

#![feature(rustc_attrs)]
#![rustc_hidden_type_of_opaques]

// Make sure that the compiler can handle `ReErased` in the hidden type of an opaque.

fn foo<'a: 'a>(x: &'a Vec<i32>) -> impl Fn() + 'static {
//~^ ERROR 0, 'a)>::{closure#0} closure_kind_ty=i8 closure_sig_as_fn_ptr_ty=extern "rust-call" fn(()) upvar_tys=()}
// Can't write whole type because of lack of path sanitization
    || ()
}

fn bar() -> impl Fn() + 'static {
//~^ ERROR , [ReErased])
// Can't write whole type because of lack of path sanitization
    foo(&vec![])
}

fn main() {}
