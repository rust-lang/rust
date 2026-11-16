fn main() {
    let number = 2;
    Some(true).filter({ //~ ERROR expected `FnOnce(&bool)` signature, found `bool`
        if number % 2 == 0 {
            number == 0
        } else {
            number != 0
        }
    });
}
