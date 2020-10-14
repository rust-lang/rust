fn main() {
    [0u8; std::mem::size_of::<_>()]; //~ ERROR: type annotations needed
}
