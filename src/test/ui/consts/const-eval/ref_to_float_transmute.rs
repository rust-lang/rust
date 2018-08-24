//compile-pass

#![feature(const_fn_union)]

fn main() {}

static FOO: u32 = 42;

union Foo {
    f: Float,
    r: &'static u32,
}

#[cfg(target_pointer_width="64")]
type Float = f64;

#[cfg(target_pointer_width="32")]
type Float = f32;

static BAR: Float = unsafe { Foo { r: &FOO }.f };
