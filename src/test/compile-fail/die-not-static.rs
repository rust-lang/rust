fn main() {
    let v = ~"test";
    let sslice = str::slice(v, 0, v.len());
    //~^ ERROR borrowed value does not live long enough
    fail!(sslice);
}
