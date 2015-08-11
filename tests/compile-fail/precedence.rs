#![feature(plugin)]
#![plugin(clippy)]

#[deny(precedence)]
#[allow(eq_op)]
fn main() {
    format!("{} vs. {}", 1 << 2 + 3, (1 << 2) + 3); //~ERROR
    format!("{} vs. {}", 1 + 2 << 3, 1 + (2 << 3)); //~ERROR
    format!("{} vs. {}", 4 >> 1 + 1, (4 >> 1) + 1); //~ERROR
    format!("{} vs. {}", 1 + 3 >> 2, 1 + (3 >> 2)); //~ERROR
    format!("{} vs. {}", 1 ^ 1 - 1, (1 ^ 1) - 1);   //~ERROR
    format!("{} vs. {}", 3 | 2 - 1, (3 | 2) - 1);   //~ERROR
    format!("{} vs. {}", 3 & 5 - 2, (3 & 5) - 2);   //~ERROR

}
