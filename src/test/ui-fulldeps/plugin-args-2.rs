// run-pass
// aux-build:plugin-args.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(plugin_args())] //~ WARNING compiler plugins are deprecated

fn main() {
    assert_eq!(plugin_args!(), "");
}
