fn main() {
    S { field ..S::default() }
    S { 0 ..S::default() }
    S { field .. }
    S { 0 .. }
}
