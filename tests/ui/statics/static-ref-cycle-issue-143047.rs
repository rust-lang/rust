// Regression test for https://github.com/rust-lang/rust/issues/143047

#[allow(static_mut_refs)]

static FOO: u8 = {
    let x = &FOO;
    //~^ ERROR: encountered static that tried to access itself during initialization
    0
};

static mut BAR: u8 = {
    let x = unsafe { &mut BAR };
    //~^ ERROR: encountered static that tried to access itself during initialization
    0
};

static mut BAZ: u8 = {
    let x: &u8 = unsafe { &*&raw const BAR };
    let y = &raw mut BAR;
    let z: &mut u8 = unsafe { &mut *y };
    let w = &z;
    let v = &w;
    0
};

static QUX: u8 = {
    //~^ ERROR: cycle detected when evaluating initializer of static `QUX`
    let x = &QUUX;
    0
};

static QUUX: u8 = {
    let x = &QUUUX;
    0
};

static QUUUX: u8 = {
    let x = &QUX;
    0
};

static PROMOTED: u8 = {
    let x = &&&PROMOTED;
    //~^ ERROR: cycle detected when evaluating initializer of static `PROMOTED`
    0
};

fn main() {}
