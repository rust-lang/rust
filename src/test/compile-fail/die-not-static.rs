// error-pattern:illegal borrow: borrowed value does not live long enough

fn main() {
    let v = ~"test";
    let sslice = str::slice(v, 0, v.len());
    fail!(sslice);
}
