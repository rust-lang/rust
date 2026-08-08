#![feature(late_bound_turbofishing)]

fn foo_early<'a: 'a>(b: &'a u32) -> &'a u32 { b }
fn foo_late<'a>(b: &'a u32) -> &'a u32 { b }
fn foo_latest(_: &u32) {}

fn require_static<T: 'static>(_: T) { }

fn main() {
    let f = foo_early::<'static>;
    require_static(f);
    let f = foo_late::<'static>;
    require_static(f);
    let f = foo_latest::<'static>;
    //~^ ERROR: function takes 0 lifetime arguments but 1 lifetime argument was supplied [E0107]
    require_static(f);
}
