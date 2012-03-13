// error-pattern:unsupported cast

fn main() {
  log(debug, 1.0 as *libc::FILE); // Can't cast float to native.
}
