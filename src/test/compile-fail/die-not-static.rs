use std::str;

fn main() {
    let v = ~"test";
    let sslice = v.slice(0, v.len());
    //~^ ERROR borrowed value does not live long enough
    fail!(sslice);
}
