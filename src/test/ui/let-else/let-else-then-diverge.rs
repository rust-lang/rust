// popped up in in #94012, where an alternative desugaring was
// causing unreachable code errors, and also we needed to check
// that the desugaring's generated lints weren't applying to
// the whole else block.

#![feature(let_else)]
#![deny(unused_variables)]
#![deny(unreachable_code)]

fn let_else_diverge() -> bool {
    let Some(_) = Some("test") else {
        let x = 5; //~ ERROR unused variable: `x`
        return false;
    };
    return true;
}

fn main() {
    let_else_diverge();
}
