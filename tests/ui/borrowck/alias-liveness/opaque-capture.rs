// check-pass

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

fn captures_temp_early<'a>(x: &'a Vec<i32>) -> impl Sized + Captures<'a> + 'static {}
fn captures_temp_late<'a: 'a>(x: &'a Vec<i32>) -> impl Sized + Captures<'a> + 'static {}

fn test() {
    let x = captures_temp_early(&vec![]);
    let y = captures_temp_late(&vec![]);
}

fn main() {}
