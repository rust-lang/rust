// Test which of the builtin types are considered sendable. The tests
// in this file all test region bound and lifetime violations that are
// detected during type check.

trait Dummy : 'static { }
fn assert_send<T:'static>() { }

// lifetime pointers with 'static lifetime are ok

fn static_lifime_ok<'a,T,U:Send>(_: &'a isize) {
    assert_send::<&'static isize>();
    assert_send::<&'static str>();
    assert_send::<&'static [isize]>();

    // whether or not they are mutable
    assert_send::<&'static mut isize>();
}

// otherwise lifetime pointers are not ok

fn param_not_ok<'a>(x: &'a isize) {
    assert_send::<&'a isize>();
    //~^ ERROR lifetime may not live long enough
}

fn param_not_ok1<'a>(_: &'a isize) {
    assert_send::<&'a str>();
    //~^ ERROR lifetime may not live long enough
}

fn param_not_ok2<'a>(_: &'a isize) {
    assert_send::<&'a [isize]>();
    //~^ ERROR lifetime may not live long enough
}

// boxes are ok

fn box_ok() {
    assert_send::<Box<isize>>();
    assert_send::<String>();
    assert_send::<Vec<isize>>();
}

// but not if they own a bad thing

fn box_with_region_not_ok<'a>() {
    assert_send::<Box<&'a isize>>();
    //~^ ERROR lifetime may not live long enough
}

// raw pointers are ok unless they point at unsendable things

fn unsafe_ok1<'a>(_: &'a isize) {
    assert_send::<*const isize>();
    assert_send::<*mut isize>();
}

fn unsafe_ok2<'a>(_: &'a isize) {
    assert_send::<*const &'a isize>();
    //~^ ERROR lifetime may not live long enough
}

fn unsafe_ok3<'a>(_: &'a isize) {
    assert_send::<*mut &'a isize>();
    //~^ ERROR lifetime may not live long enough
}

fn main() {
}
