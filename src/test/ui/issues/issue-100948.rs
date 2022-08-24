// compile-flags: -Zunpretty=hir

fn main() {
    1u; //~ ERROR invalid suffix `u` for number literal
}
