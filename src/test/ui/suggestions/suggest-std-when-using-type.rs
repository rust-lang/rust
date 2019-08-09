fn main() {
    let pi = f32::consts::PI; //~ ERROR ambiguous associated type
    let bytes = "hello world".as_bytes();
    let string = unsafe {
        str::from_utf8(bytes) //~ ERROR no function or associated item named `from_utf8` found
    };
}
