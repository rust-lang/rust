// Don't leak the unique pointers

resource r(v: *int) unsafe {
    let v2: ~int = unsafe::reinterpret_cast(v);
}

enum t = {
    mut next: option<@t>,
    r: r
};

fn main() unsafe {
    let i1 = ~0;
    let i1p = unsafe::reinterpret_cast(i1);
    unsafe::forget(i1);
    let i2 = ~0;
    let i2p = unsafe::reinterpret_cast(i2);
    unsafe::forget(i2);

    let x1 = @t({
        mut next: none,
        r: r(i1p)
    });
    let x2 = @t({
        mut next: none,
        r: r(i2p)
    });
    x1.next = some(x2);
    x2.next = some(x1);
}
