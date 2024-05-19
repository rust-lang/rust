// Ensure that temporaries in if-let guards live for the arm
// regression test for #118593

//@ check-pass

#![feature(if_let_guard)]
#![feature(let_chains)]

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

fn let_let_chain_guard(num: u8) {
    match num {
        5 | 6
            if let Some(ref a) = get_temp()
                && let Some(ref b) = get_temp() =>
        {
            let _x = a;
            let _y = b;
        }
        _ => {}
    }
    match num {
        7 | 8
            if let Some(ref mut c) = get_temp()
                && let Some(ref mut d) = get_temp() =>
        {
            let _w = c;
            let _z = d;
        }
        _ => {}
    }
}

fn let_cond_chain_guard(num: u8) {
    match num {
        9 | 10
            if let Some(ref a) = get_temp()
                && true =>
        {
            let _x = a;
        }
        _ => {}
    }
    match num {
        11 | 12
            if let Some(ref mut b) = get_temp()
                && true =>
        {
            let _w = b;
        }
        _ => {}
    }
}

fn main() {}
