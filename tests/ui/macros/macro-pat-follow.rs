// run-pass
macro_rules! pat_in {
    ($p:pat in $e:expr) => {{
        let mut iter = $e.into_iter();
        while let $p = iter.next() {}
    }};
}

macro_rules! pat_if {
    ($p:pat if $e:expr) => {{
        match Some(1u8) {
            $p if $e => {}
            _ => {}
        }
    }};
}

fn main() {
    pat_in!(Some(_) in 0..10);
    pat_if!(Some(x) if x > 0);
}
