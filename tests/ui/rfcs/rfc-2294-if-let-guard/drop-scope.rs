// Ensure that temporaries in if-let guards live for the arm
// regression test for #118593

//@ check-pass

#![feature(if_let_guard)]

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
