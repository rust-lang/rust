fn require_sync<T: Sync>() {}
//~^ NOTE required by this bound in `require_sync`
//~| NOTE required by this bound in `require_sync`
//~| NOTE required by this bound in `require_sync`
//~| NOTE required by this bound in `require_sync`
//~| NOTE required by a bound in `require_sync`
//~| NOTE required by a bound in `require_sync`
//~| NOTE required by a bound in `require_sync`
//~| NOTE required by a bound in `require_sync`

fn main() {
    require_sync::<std::cell::Cell<()>>();
    //~^ ERROR `Cell<()>` cannot be shared between threads safely
    //~| NOTE `Cell<()>` cannot be shared between threads safely
    //~| NOTE if you want to do aliasing and mutation between multiple threads, use `std::sync::RwLock`

    require_sync::<std::cell::Cell<u8>>();
    //~^ ERROR `Cell<u8>` cannot be shared between threads safely
    //~| NOTE `Cell<u8>` cannot be shared between threads safely
    //~| NOTE if you want to do aliasing and mutation between multiple threads, use `std::sync::RwLock` or `std::sync::atomic::AtomicU8` instead

    require_sync::<std::cell::Cell<i32>>();
    //~^ ERROR `Cell<i32>` cannot be shared between threads safely
    //~| NOTE `Cell<i32>` cannot be shared between threads safely
    //~| NOTE if you want to do aliasing and mutation between multiple threads, use `std::sync::RwLock` or `std::sync::atomic::AtomicI32` instead

    require_sync::<std::cell::Cell<bool>>();
    //~^ ERROR `Cell<bool>` cannot be shared between threads safely
    //~| NOTE `Cell<bool>` cannot be shared between threads safely
    //~| NOTE if you want to do aliasing and mutation between multiple threads, use `std::sync::RwLock` or `std::sync::atomic::AtomicBool` instead
}
