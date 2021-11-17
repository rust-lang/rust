fn static_id<'a,'b>(t: &'a ()) -> &'static ()
    where 'a: 'static { t }
fn static_id_indirect<'a,'b>(t: &'a ()) -> &'static ()
    where 'a: 'b, 'b: 'static { t }
fn static_id_wrong_way<'a>(t: &'a ()) -> &'static () where 'static: 'a {
    t //~ ERROR E0312
}

fn error(u: &(), v: &()) {
    static_id(&u); //~ ERROR `u` has an anonymous lifetime `'_` but it needs to satisfy a `'static` lifetime requirement [E0759]
    static_id_indirect(&v); //~ ERROR `v` has an anonymous lifetime `'_` but it needs to satisfy a `'static` lifetime requirement [E0759]
}

fn main() {}
