// xfail-fast
#[legacy_modes];
#[abi = "rust-intrinsic"]
extern mod rusti {
    #[legacy_exports];
    fn frame_address(f: fn(*u8));
}

fn main() {
    do rusti::frame_address |addr| {
        assert addr.is_not_null();
    }
}
