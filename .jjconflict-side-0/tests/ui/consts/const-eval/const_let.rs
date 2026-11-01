fn main() {}

struct FakeNeedsDrop;

impl Drop for FakeNeedsDrop {
    fn drop(&mut self) {}
}

// ok
const X: FakeNeedsDrop = { let x = FakeNeedsDrop; x };

// ok (used to incorrectly error, see #62273)
const X2: FakeNeedsDrop = { let x; x = FakeNeedsDrop; x };

// error
const Y: FakeNeedsDrop = { let mut x = FakeNeedsDrop; x = FakeNeedsDrop; x };
//~^ ERROR destructor of

// error
const Y2: FakeNeedsDrop = { let mut x; x = FakeNeedsDrop; x = FakeNeedsDrop; x };
//~^ ERROR destructor of

// error
const Z: () = { let mut x = None; x = Some(FakeNeedsDrop); };
//~^ ERROR destructor of

// error
const Z2: () = { let mut x; x = None; x = Some(FakeNeedsDrop); };
//~^ ERROR destructor of
