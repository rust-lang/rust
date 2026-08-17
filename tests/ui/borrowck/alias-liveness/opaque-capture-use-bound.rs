//@ check-pass

fn captures_temp_late<'a>(x: &'a Vec<i32>) -> impl Sized + use<'a> + 'static {}
fn captures_temp_early<'a: 'a>(x: &'a Vec<i32>) -> impl Sized + use<'a> + 'static {}

fn test() {
    let x = captures_temp_early(&vec![]);
    let y = captures_temp_late(&vec![]);
}

fn main() {}
