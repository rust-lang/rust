#[abi = "rust-intrinsic"]
native mod rusti {
    fn frame_address() -> *u8;
}

fn main() {
    assert rusti::frame_address().is_not_null();
}
