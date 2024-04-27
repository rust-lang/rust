//@ run-pass
//@ edition:2018

macro_rules! pat_bar {
    ($p:pat | $p2:pat) => {{
        match Some(1u8) {
            $p | $p2 => {}
            _ => {}
        }
    }};
}

fn main() {
    pat_bar!(Some(1u8) | None);
}
