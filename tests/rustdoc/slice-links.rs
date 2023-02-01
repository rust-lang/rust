#![crate_name = "foo"]
#![no_std]

pub struct MyBox<T: ?Sized>(*const T);

// @has 'foo/fn.alpha.html'
// @snapshot link_slice_u32 - '//pre[@class="rust item-decl"]/code'
pub fn alpha() -> &'static [u32] {
    loop {}
}

// @has 'foo/fn.beta.html'
// @snapshot link_slice_generic - '//pre[@class="rust item-decl"]/code'
pub fn beta<T>() -> &'static [T] {
    loop {}
}

// @has 'foo/fn.gamma.html'
// @snapshot link_box_u32 - '//pre[@class="rust item-decl"]/code'
pub fn gamma() -> MyBox<[u32]> {
    loop {}
}

// @has 'foo/fn.delta.html'
// @snapshot link_box_generic - '//pre[@class="rust item-decl"]/code'
pub fn delta<T>() -> MyBox<[T]> {
    loop {}
}
