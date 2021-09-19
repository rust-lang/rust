// Test that changing what a `type` points to does not go unnoticed.

// incremental
// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]
#![allow(dead_code)]
#![allow(unused_variables)]

fn main() { }


#[rustc_if_this_changed]
type TypeAlias = u32;

// The type alias directly affects the type of the field,
// not the enclosing struct:
#[rustc_then_this_would_need(type_of)] //~ ERROR no path
struct Struct {
    #[rustc_then_this_would_need(type_of)] //~ ERROR OK
    x: TypeAlias,
    y: u32
}

#[rustc_then_this_would_need(type_of)] //~ ERROR no path
enum Enum {
    Variant1 {
        #[rustc_then_this_would_need(type_of)] //~ ERROR OK
        t: TypeAlias
    },
    Variant2(i32)
}

#[rustc_then_this_would_need(type_of)] //~ ERROR no path
trait Trait {
    #[rustc_then_this_would_need(fn_sig)] //~ ERROR OK
    fn method(&self, _: TypeAlias);
}

struct SomeType;

#[rustc_then_this_would_need(type_of)] //~ ERROR no path
impl SomeType {
    #[rustc_then_this_would_need(fn_sig)] //~ ERROR OK
    #[rustc_then_this_would_need(typeck)] //~ ERROR OK
    fn method(&self, _: TypeAlias) {}
}

#[rustc_then_this_would_need(type_of)] //~ ERROR OK
type TypeAlias2 = TypeAlias;

#[rustc_then_this_would_need(fn_sig)] //~ ERROR OK
#[rustc_then_this_would_need(typeck)] //~ ERROR OK
fn function(_: TypeAlias) {

}
