// ignore-windows: TODO the windows hook is not done yet
// compile-flags: -Zmiri-disable-isolation

fn main() {
    std::env::current_dir().unwrap();
}
