// Test that `impl Alpha<dyn Object>` resets the object-lifetime
// default to `'static`.
//
//@ check-pass

trait Alpha<Item: ?Sized> {
    fn item(&self) -> Box<Item> {
        panic!()
    }
}

trait Object {}
impl<T> Alpha<dyn Object> for T {}
fn alpha(x: &str, y: &str) -> impl Alpha<dyn Object> { () }
fn is_static<T>(_: T) where T: 'static { }

fn bar(x: &str) -> &impl Alpha<dyn Object> { &() }

fn main() {
    let s = format!("foo");
    let r = bar(&s);
    is_static(r.item());
}
