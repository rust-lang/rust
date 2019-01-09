// aux-build:plugin_args.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(plugin_args())]

fn main() {
    assert_eq!(plugin_args!(), "");
}
