// Test which of the builtin types are considered sendable.

fn assert_send<T:Send>() { }

// owned content are ok
fn test30() { assert_send::<Box<isize>>(); }
fn test31() { assert_send::<String>(); }
fn test32() { assert_send::<Vec<isize> >(); }

// but not if they own a bad thing
fn test40() {
    assert_send::<Box<*mut u8>>();
    //~^ ERROR `*mut u8` cannot be sent between threads safely
}

fn main() { }
