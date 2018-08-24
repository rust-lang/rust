#![feature(rustc_attrs)]
#![allow(dead_code)]

trait RegularExpression: Sized {
    type Text;
}

struct ExecNoSyncStr<'a>(&'a u8);

impl<'c> RegularExpression for ExecNoSyncStr<'c> {
    type Text = u8;
}

struct FindCaptures<'t, R>(&'t R::Text) where R: RegularExpression, R::Text: 't;

enum FindCapturesInner<'r, 't> {
    Dynamic(FindCaptures<'t, ExecNoSyncStr<'r>>),
}

#[rustc_error]
fn main() {}    //~ ERROR compilation successful
