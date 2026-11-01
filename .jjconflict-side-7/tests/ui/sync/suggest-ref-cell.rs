fn require_sync<T: Sync>() {}
//~^ NOTE required by this bound in `require_sync`
//~| NOTE required by a bound in `require_sync`

fn main() {
    require_sync::<std::cell::RefCell<()>>();
    //~^ ERROR `RefCell<()>` cannot be shared between threads safely
    //~| NOTE `RefCell<()>` cannot be shared between threads safely
    //~| NOTE use `std::sync::RwLock` instead
}
