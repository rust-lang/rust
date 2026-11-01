//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no
//@ normalize-stderr: "DefId\((\d+):(\d+)" -> "DefId(..:.."
//@ normalize-stderr: "\[[A-Fa-f0-9]{4}\]" -> "[....]"

#![feature(negative_impls)]
#![feature(rustc_attrs)]

#[rustc_dump_clauses]
fn function<T: Send>(_t: T) {}

#[rustc_dump_clauses]
trait Trait: Sync {
    #[rustc_dump_clauses]
    type Assoc;
}

#[rustc_dump_clauses]
struct X<'a, T: ?Sized, I: Iterator> {
    x: &'a T,
    y: &'a I::Item,
}

#[rustc_dump_clauses]
impl<T: ?Sized, I> !Sync for X<'_, T, I> {}
