// See https://github.com/rust-lang/rust/pull/67911#issuecomment-576023915
fn f<'a, 'b>(x: i32) -> (&'a i32, &'b i32) {
    let y = &x;
    (y, y) //~ ERROR cannot return
}

fn main() {}
