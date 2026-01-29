//@ run-pass

pub fn main() {
    /*
    let _: Vec<Box<dyn Fn(isize) -> _>> = vec![
        Box::new(|x| (x as u8)),
        Box::new(|x| (x as i16 as u8)),
    ];
    */
    let _: &[Box<dyn Fn(isize) -> _>] = &[Box::new(|x| x as u8), Box::new(|x| x as i16 as u8)];
}
