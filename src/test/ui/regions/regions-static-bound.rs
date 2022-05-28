// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

fn static_id<'a,'b>(t: &'a ()) -> &'static ()
    where 'a: 'static { t }
//~^ WARN unnecessary lifetime parameter `'a`

fn static_id_indirect<'a,'b>(t: &'a ()) -> &'static ()
    where 'a: 'b, 'b: 'static { t }
//~^ WARN unnecessary lifetime parameter `'b`

fn static_id_wrong_way<'a>(t: &'a ()) -> &'static () where 'static: 'a {
    t
    //[base]~^ ERROR E0312
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn error(u: &(), v: &()) {
    static_id(&u);
    //[base]~^ ERROR `u` has an anonymous lifetime `'_` but it needs to satisfy a `'static` lifetime requirement [E0759]
    //[nll]~^^ ERROR borrowed data escapes outside of function [E0521]
    static_id_indirect(&v);
    //[base]~^ ERROR `v` has an anonymous lifetime `'_` but it needs to satisfy a `'static` lifetime requirement [E0759]
    //[nll]~^^ ERROR borrowed data escapes outside of function [E0521]
}

fn main() {}
