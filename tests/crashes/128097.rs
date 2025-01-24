//@ known-bug: #128097
#![feature(explicit_tail_calls)]
fn f(x: &mut ()) {
    let _y: String;
    become f(x);
}
