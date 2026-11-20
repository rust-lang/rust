// Test which trait objects and basic types are considered sendable, considering lifetimes.

fn assert_send_static<T: Send + 'static>() {}
fn assert_send<T: Send>() {}

trait Dummy {}

fn test1<'a>() {
    assert_send_static::<&'a dyn Dummy>();
    //~^ ERROR `&'a (dyn Dummy + 'a)` cannot be sent between threads safely [E0277]
}

fn test2<'a>() {
    assert_send_static::<&'a (dyn Dummy + Sync)>();
    //~^ ERROR: lifetime may not live long enough
}

fn test3<'a>() {
    assert_send_static::<Box<dyn Dummy + 'a>>();
    //~^ ERROR `(dyn Dummy + 'a)` cannot be sent between threads safely
}

fn test4<'a>() {
    assert_send::<*mut &'a isize>();
    //~^ ERROR `*mut &'a isize` cannot be sent between threads safely
}

fn main() {
    assert_send_static::<&'static (dyn Dummy + Sync)>();
    assert_send_static::<Box<dyn Dummy + Send>>();

    assert_send::<&'static dyn Dummy>();
    //~^ ERROR `&'static (dyn Dummy + 'static)` cannot be sent between threads safely [E0277]
    assert_send::<Box<dyn Dummy>>();
    //~^ ERROR `dyn Dummy` cannot be sent between threads safely
    assert_send::<&'static (dyn Dummy + Sync)>();
    assert_send::<Box<dyn Dummy + Send>>();

    // owned content is ok
    assert_send::<Box<isize>>();
    assert_send::<String>();
    assert_send::<Vec<isize>>();

    // but not if it owns a bad thing
    assert_send::<Box<*mut u8>>();
    //~^ ERROR `*mut u8` cannot be sent between threads safely

    assert_send::<*mut isize>();
    //~^ ERROR `*mut isize` cannot be sent between threads safely
}

fn object_ref_with_static_bound_not_ok() {
    assert_send::<&'static (dyn Dummy + 'static)>();
    //~^ ERROR `&'static (dyn Dummy + 'static)` cannot be sent between threads safely [E0277]
}
