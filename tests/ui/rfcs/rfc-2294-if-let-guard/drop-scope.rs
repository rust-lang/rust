// Ensure that temporaries in if-let guards live for the arm
// regression test for #118593

//@ check-pass
//@revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024

fn get_temp() -> Option<String> {
    None
}

fn let_guard(num: u8) {
    match num {
        1 | 2 if let Some(ref a) = get_temp() => {
            let _b = a;
        }
        _ => {}
    }
    match num {
        3 | 4 if let Some(ref mut c) = get_temp() => {
            let _d = c;
        }
        _ => {}
    }
}

fn main() {}
