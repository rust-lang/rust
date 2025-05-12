enum NonAscii {
    Abcd,
    Éfgh,
}

use NonAscii::*;

fn f(x: NonAscii) -> bool {
    match x {
        Éfgh => true,
        _ => false,
    }
}

fn main() {
    dbg!(f(Abcd));
}
