// Regression test for issue #48238

fn use_val<'a>(val: &'a u8) -> &'a u8 {
    val
}

fn main() {
    let orig: u8 = 5;
    move || use_val(&orig); //~ ERROR
}
