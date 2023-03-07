#![warn(unused_lifetimes)]

fn static_id<'a,'b>(t: &'a ()) -> &'static () where 'a: 'static { t }
//~^ WARN lifetime parameter `'b` never used
//~| WARN unnecessary lifetime parameter `'a`

fn static_id_indirect<'a,'b>(t: &'a ()) -> &'static ()
    where 'a: 'b, 'b: 'static { t }
//~^ WARN unnecessary lifetime parameter `'b`

fn static_id_wrong_way<'a>(t: &'a ()) -> &'static () where 'static: 'a {
    t
    //~^ ERROR lifetime may not live long enough
}

fn error(u: &(), v: &()) {
    static_id(&u);
    //~^ ERROR borrowed data escapes outside of function [E0521]
    static_id_indirect(&v);
    //~^ ERROR borrowed data escapes outside of function [E0521]
}

fn main() {}
