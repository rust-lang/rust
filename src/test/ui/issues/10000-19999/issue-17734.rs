// run-pass
// Test that generating drop glue for Box<str> doesn't ICE


fn f(s: Box<str>) -> Box<str> {
    s
}

fn main() {
    // There is currently no safe way to construct a `Box<str>`, so improvise
    let box_arr: Box<[u8]> = Box::new(['h' as u8, 'e' as u8, 'l' as u8, 'l' as u8, 'o' as u8]);
    let box_str: Box<str> = unsafe { std::mem::transmute(box_arr) };
    assert_eq!(&*box_str, "hello");
    f(box_str);
}
