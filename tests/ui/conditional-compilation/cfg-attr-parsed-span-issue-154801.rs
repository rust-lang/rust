fn main() {
    let _x = 30;
    #[cfg_attr(, (cc))] //~ ERROR expected a literal (`1u8`, `1.0f32`, `"string"`, etc.) here, found `,`
    _x //~ ERROR mismatched types
}

fn inline_case() {
    let _x = 30;
    #[inline] //~ ERROR `#[inline]` attribute cannot be used on expressions
    _x //~ ERROR mismatched types
}
