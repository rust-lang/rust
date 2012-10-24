fn main() {
    let a = ~[1, 2, 3, 4, 5];
    let mut b = ~[a, a];
    b = b + b; // FIXME(#3387)---can't write b += b
}
