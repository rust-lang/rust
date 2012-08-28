#[nolink]
#[abi = "rust-intrinsic"]
extern mod rusti {
    fn morestack_addr() -> *();
}

fn main() {
    let addr = rusti::morestack_addr();
    assert addr.is_not_null();
    error!("%?", addr);
}