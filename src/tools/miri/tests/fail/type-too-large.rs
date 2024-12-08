//@ignore-bitwidth: 32

fn main() {
    let _fat: [u8; (1 << 61) + (1 << 31)]; // ideally we'd error here, but we avoid computing the layout until absolutely necessary
    _fat = [0; (1u64 << 61) as usize + (1u64 << 31) as usize]; //~ ERROR: post-monomorphization error
}
