macro_rules! prob1 {
    (0) => {
        0
    };
    ($n:expr) => {
        if ($n % 3 == 0) || ($n % 5 == 0) {
            $n + prob1!($n - 1); //~ ERROR recursion limit reached while expanding `prob1!`
        } else {
            prob1!($n - 1);
        }
    };
}

fn main() {
    println!("Problem 1: {}", prob1!(1000));
}
