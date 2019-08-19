// Test which object types are considered sendable. This test
// is broken into two parts because some errors occur in distinct
// phases in the compiler. See kindck-send-object2.rs as well!

fn assert_send<T:Send+'static>() { }
trait Dummy { }

// careful with object types, who knows what they close over...
fn test51<'a>() {
    assert_send::<&'a dyn Dummy>();
    //~^ ERROR `(dyn Dummy + 'a)` cannot be shared between threads safely [E0277]
}
fn test52<'a>() {
    assert_send::<&'a (dyn Dummy + Sync)>();
    //~^ ERROR does not fulfill the required lifetime
}

// ...unless they are properly bounded
fn test60() {
    assert_send::<&'static (dyn Dummy + Sync)>();
}
fn test61() {
    assert_send::<Box<dyn Dummy + Send>>();
}

// closure and object types can have lifetime bounds which make
// them not ok
fn test_71<'a>() {
    assert_send::<Box<dyn Dummy + 'a>>();
    //~^ ERROR `(dyn Dummy + 'a)` cannot be sent between threads safely
}

fn main() { }
