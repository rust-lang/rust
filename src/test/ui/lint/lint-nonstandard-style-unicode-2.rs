#![allow(dead_code)]

#![forbid(non_snake_case)]

// Some scripts (e.g., hiragana) don't have a concept of
// upper/lowercase

// 2. non_snake_case

// Can only use non-uppercase letters.
// So this works:

fn 编程() {}

// but this doesn't:

fn Ц() {}
//~^ ERROR function `Ц` should have a snake case name

// besides this, you cannot use continuous underscores in the middle

fn 分__隔() {}
//~^ ERROR function `分__隔` should have a snake case name

// but you can use them both at the beginning and at the end.

fn _______不_连_续_的_存_在_______() {}

fn main() {}
