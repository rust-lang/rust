fn require_send<T: Send>(_: T) {}

fn main() {
    let foo = std::ptr::null_mut::<()>();
    let other = String::new();

    require_send(move || {
        //~^ ERROR `*mut ()` cannot be sent between threads safely
        drop(other);
        unsafe { foo.read() };
    });
}
