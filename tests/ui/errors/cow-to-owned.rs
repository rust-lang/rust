// issue #144792

fn main() {
    _ = std::env::var_os("RUST_LOG").map_or("warn".into(), |x| x.to_string_lossy().to_owned());
    //~^ ERROR cannot return value referencing function parameter
    //~| HELP try using `.into_owned()`
}
