fn require_sync<T: Sync>() {}
//~^ NOTE required by this bound in `require_sync`
//~| NOTE required by a bound in `require_sync`

fn main() {
    require_sync::<std::cell::OnceCell<()>>();
    //~^ ERROR `OnceCell<()>` cannot be shared between threads safely
    //~| NOTE `OnceCell<()>` cannot be shared between threads safely
    //~| NOTE use `std::sync::OnceLock` instead
}
