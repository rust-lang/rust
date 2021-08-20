// compile-flags: --remap-path-prefix={{src-base}}=remapped

fn main() {
    ferris //~ ERROR cannot find value `ferris` in this scope
}
