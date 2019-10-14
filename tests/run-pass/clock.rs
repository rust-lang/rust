// ignore-windows: TODO clock shims are not implemented on Windows
// compile-flags: -Zmiri-disable-isolation

use std::time::SystemTime;

fn main() {
   let _now = SystemTime::now();
}
