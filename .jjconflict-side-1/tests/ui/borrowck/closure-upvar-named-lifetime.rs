// Regression test for #153545.
//
// When a closure captures a variable whose type involves a named
// lifetime from the parent function, diagnostics should use the
// actual lifetime name (e.g., `'a`) instead of a synthetic name
// like `'1`.

use std::cell::RefCell;
use std::sync::Arc;
use std::collections::HashMap;
use std::collections::hash_map::Entry;

fn apply<'a>(
    f: Arc<dyn Fn(Entry<'a, String, String>) + 'a>,
) -> impl Fn(RefCell<HashMap<String, String>>)
{
    move |map| {
    //~^ ERROR hidden type for `impl Fn(RefCell<HashMap<String, String>>)` captures lifetime that does not appear in bounds
        let value = map.borrow_mut().entry("foo".to_string());
        //~^ ERROR `map` does not live long enough
        //~| ERROR temporary value dropped while borrowed
        let wrapped_value = value;
        f(wrapped_value);
    }
}

// Also test with Box instead of Arc, a simpler captured type.
fn apply_box<'a>(
    f: Box<dyn Fn(Entry<'a, String, String>) + 'a>,
) -> impl Fn(RefCell<HashMap<String, String>>)
{
    move |map| {
    //~^ ERROR hidden type for `impl Fn(RefCell<HashMap<String, String>>)` captures lifetime that does not appear in bounds
        let value = map.borrow_mut().entry("foo".to_string());
        //~^ ERROR `map` does not live long enough
        //~| ERROR temporary value dropped while borrowed
        f(value);
    }
}

fn main() {}
