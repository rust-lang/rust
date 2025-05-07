#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct Foo {
    val16: u16,
    // Padding bytes go here!
    val32: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct Bar {
    bytes: [u8; 8],
}

#[repr(C)]
union FooBar {
    foo: Foo,
    bar: Bar,
}

pub fn main() {
    // Initialize as u8 to ensure padding bytes are zeroed.
    let mut foobar = FooBar { bar: Bar { bytes: [0u8; 8] } };
    // Reading either field is ok.
    let _val = unsafe { (foobar.foo, foobar.bar) };
    // Does this assignment copy the uninitialized padding bytes
    // over the initialized padding bytes? miri doesn't seem to think so.
    foobar.foo = Foo { val16: 1, val32: 2 };
    // This resets the padding to uninit.
    let _val = unsafe { (foobar.foo, foobar.bar) };
    //~^ ERROR: uninitialized
}
