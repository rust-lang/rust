#![feature(optin_builtin_traits)]

struct MySendable {
   t: *mut u8
}

unsafe impl Send for MySendable {}

struct MyNotSendable {
   t: *mut u8
}

impl !Send for MyNotSendable {}

fn is_send<T: Send>() {}

fn main() {
    is_send::<MySendable>();
    is_send::<MyNotSendable>();
    //~^ ERROR `MyNotSendable` cannot be sent between threads safely
}
