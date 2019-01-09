#![feature(extern_in_paths)]

fn main() {
    let s = extern::xcrate::S;
    //~^ ERROR failed to resolve: could not find `xcrate` in `extern`
}
