//@ run-rustfix

fn main() {
    let x = std::sync::Mutex::new(1usize);
    x.lock().unwrap() = 2;
    //~^ ERROR invalid left-hand side of assignment
    x.lock().unwrap() += 1;
    //~^ ERROR binary assignment operation `+=` cannot be applied to type `std::sync::MutexGuard<'_, usize>`

    let mut y = x.lock().unwrap();
    y = 2;
    //~^ ERROR mismatched types
    y += 1;
    //~^ ERROR binary assignment operation `+=` cannot be applied to type `std::sync::MutexGuard<'_, usize>`
}
