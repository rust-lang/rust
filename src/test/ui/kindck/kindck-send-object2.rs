// Continue kindck-send-object1.rs.

fn assert_send<T:Send>() { }
trait Dummy { }

fn test50() {
    assert_send::<&'static Dummy>();
    //~^ ERROR `(dyn Dummy + 'static)` cannot be shared between threads safely [E0277]
}

fn test53() {
    assert_send::<Box<Dummy>>();
    //~^ ERROR `dyn Dummy` cannot be sent between threads safely
}

// ...unless they are properly bounded
fn test60() {
    assert_send::<&'static (Dummy+Sync)>();
}
fn test61() {
    assert_send::<Box<Dummy+Send>>();
}

fn main() { }
