//! Check that destructors of main thread thread locals are executed when main returns.
//! Only works on tagets that use something other than pthread keys for the destructors.
//@ignore-target: freebsd illumos solaris android
//@run-native
//@revisions: with_exit without_exit

struct Bar;

impl Drop for Bar {
    fn drop(&mut self) {
        println!("Bar dtor");
    }
}

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
        println!("Foo dtor");
        // We initialize another thread-local inside the dtor, which is an interesting corner case.
        // Also we use a `const` thread-local here, just to also have that code path covered.
        thread_local!(static BAR: Bar = const { Bar });
        BAR.with(|_| {});
    }
}

thread_local!(static FOO: Foo = Foo);

fn main() {
    FOO.with(|_| {});
    if cfg!(with_exit) {
        std::process::exit(0);
    }
}
