fn main() {}

struct FakeNeedsDrop;

impl Drop for FakeNeedsDrop {
    fn drop(&mut self) {}
}

// ok
const X: FakeNeedsDrop = { let x = FakeNeedsDrop; x };

// error
const Y: FakeNeedsDrop = { let mut x = FakeNeedsDrop; x = FakeNeedsDrop; x };
//~^ ERROR constant contains unimplemented expression type

// error
const Z: () = { let mut x = None; x = Some(FakeNeedsDrop); };
//~^ ERROR constant contains unimplemented expression type
