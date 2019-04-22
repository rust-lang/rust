// revisions: migrate nll
//[nll] compile-flags:-Zborrowck=mir

fn static_id<'a,'b>(t: &'a ()) -> &'static ()
    where 'a: 'static { t }
fn static_id_indirect<'a,'b>(t: &'a ()) -> &'static ()
    where 'a: 'b, 'b: 'static { t }
fn static_id_wrong_way<'a>(t: &'a ()) -> &'static () where 'static: 'a {
    t //[migrate]~ ERROR E0312
        //[nll]~^ ERROR lifetime may not live long enough
}

fn error(u: &(), v: &()) {
    static_id(&u); //[migrate]~ ERROR explicit lifetime required in the type of `u` [E0621]
    //[nll]~^ ERROR explicit lifetime required in the type of `u` [E0621]
    static_id_indirect(&v); //[migrate]~ ERROR explicit lifetime required in the type of `v` [E0621]
    //[nll]~^ ERROR explicit lifetime required in the type of `v` [E0621]
}

fn main() {}
