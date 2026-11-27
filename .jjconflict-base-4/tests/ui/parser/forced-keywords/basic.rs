//@ edition: 2021..
//@ check-pass
#![feature(forced_keywords)]

k#mod module {
    k#pub(k#in k#super) k#static _DATA: i32 = 0i8 k#as k#_;
}

k#use ::std::process::Termination;

k#fn main() -> k#impl k#self::Termination {
    k#let k#true = k#false k#else { k#return };

    k#let k#ref k#mut _x: ();

    k#const k#fn perform() -> k#impl Sized {
        k#loop {
            k#break k#match () { () k#if k#true => {} k#_ => {} };
        }
    }

    perform();
}
