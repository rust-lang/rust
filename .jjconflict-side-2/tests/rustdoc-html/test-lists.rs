#![crate_name = "foo"]

//@ has foo/fn.f.html
//@ has - //ol/li "list"
//@ has - //ol/li/ol/li "fooooo"
//@ has - //ol/li/ol/li "x"
//@ has - //ol/li "foo"
/// 1. list
///     1. fooooo
///     2. x
/// 2. foo
pub fn f() {}

//@ has foo/fn.foo2.html
//@ has - //ul/li "normal list"
//@ has - //ul/li/ul/li "sub list"
//@ has - //ul/li/ul/li "new elem still same elem and again same elem!"
//@ has - //ul/li "new big elem"
/// * normal list
///     * sub list
///     * new elem
///       still same elem
///
///       and again same elem!
/// * new big elem
pub fn foo2() {}
