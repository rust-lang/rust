//@no-rustfix: `is_ok` and `is_err` can change significant drop order
#![warn(clippy::unnecessary_map_or)]

fn main() {
    let mutex = std::sync::Mutex::new(());

    let result = Ok::<_, ()>(mutex.lock().unwrap());
    let _ = result.map_or(false, |_| true);
    //~^ unnecessary_map_or

    let result = Err::<(), _>(mutex.lock().unwrap());
    let _ = result.map_or_else(|_| true, |_| false);
    //~^ unnecessary_map_or
}
