#[repr(transparent)]
union OkButUnstableUnion { //~ ERROR transparent unions are unstable
    field: u8,
    zst: (),
}

fn main() {}
