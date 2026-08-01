#[repr(complex)] //~ ERROR: `repr(complex)` is experimental
struct Named {
    a: u64,
    b: u64,
}

#[repr(complex)] //~ ERROR: `repr(complex)` is experimental
struct Unnamed(u64, u64);

#[repr(C)]
//~^ ERROR conflicting representation hints
//~| WARN this was previously accepted
#[repr(complex)] //~ ERROR: `repr(complex)` is experimental
struct ConflictingRepr(u64, u64);

#[repr(complex)]
//~^ ERROR: `repr(complex)` is experimental
//~| ERROR: attribute cannot be used on
union U {
    f: u32,
}

#[repr(complex)]
//~^ ERROR: `repr(complex)` is experimental
//~| error: attribute cannot be used on
enum E {
    X,
}

fn main() {}
