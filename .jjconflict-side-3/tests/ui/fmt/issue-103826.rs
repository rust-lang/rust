fn main() {
    format!("{\x7D");
    //~^ ERROR 1 positional argument in format string, but no arguments were given
    format!("\x7B\x7D");
    //~^ ERROR 1 positional argument in format string, but no arguments were given
    format!("{\x7D {\x7D");
    //~^ ERROR 2 positional arguments in format string, but no arguments were given
}
