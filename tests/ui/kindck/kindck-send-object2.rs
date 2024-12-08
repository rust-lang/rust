// Continue kindck-send-object1.rs.

fn assert_send<T:Send>() { }
trait Dummy { }

fn test50() {
    assert_send::<&'static dyn Dummy>();
    //~^ ERROR `&'static (dyn Dummy + 'static)` cannot be sent between threads safely [E0277]
}

fn test53() {
    assert_send::<Box<dyn Dummy>>();
    //~^ ERROR `dyn Dummy` cannot be sent between threads safely
}

// ...unless they are properly bounded
fn test60() {
    assert_send::<&'static (dyn Dummy + Sync)>();
}
fn test61() {
    assert_send::<Box<dyn Dummy + Send>>();
}

fn main() { }
