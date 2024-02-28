//@ ignore-wasm
//@ ignore-msvc
//@ ignore-emscripten
//@ ignore-uwp

fn main() {
    let data: &[u8] = &[0; 10];
    let _: &[i8] = data.into();
    //~^ ERROR trait `From<&[u8]>` is not implemented for `&[i8]`
}
