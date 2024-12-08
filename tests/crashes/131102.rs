//@ known-bug: #131102
pub struct Blorb<const N: u16>([String; N]);
pub struct Wrap(Blorb<0>);
pub const fn i(_: Wrap) {}
