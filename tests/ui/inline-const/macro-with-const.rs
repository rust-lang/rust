// check-pass

macro_rules! exp {
    (const $n:expr) => {
        $n
    };
}

macro_rules! stmt {
    (exp $e:expr) => {
        $e
    };
    (exp $($t:tt)+) => {
        exp!($($t)+)
    };
}

fn main() {
    stmt!(exp const 1);
}
