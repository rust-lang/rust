// This is a regression test for <https://github.com/rust-lang/rust/issues/106226>.
type F = [_; ()];
//~^ ERROR
