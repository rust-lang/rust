//@ known-bug: #150040

fn main() {
    let [(ref a, b), x];
    a = "";
    b = 5;
}
