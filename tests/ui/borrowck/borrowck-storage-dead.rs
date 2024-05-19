fn ok() {
    loop {
        let _x = 1;
    }
}

fn also_ok() {
    loop {
        let _x = String::new();
    }
}

fn fail() {
    loop {
        let x: i32;
        let _ = x + 1; //~ERROR [E0381]
    }
}

fn main() {
    ok();
    also_ok();
    fail();
}
