#![feature(builtin_syntax)]

fn main() {
    builtin # foobar(); //~ ERROR unknown `builtin #` construct
}

fn not_identifier() {
    builtin # {}(); //~ ERROR expected identifier after
}
