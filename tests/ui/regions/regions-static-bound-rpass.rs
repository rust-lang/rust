//@ run-pass

#![warn(redundant_lifetimes)]

fn invariant_id<'a,'b>(t: &'b mut &'static ()) -> &'b mut &'a ()
//~^ WARN unnecessary lifetime parameter `'a`
    where 'a: 'static { t }

fn static_id<'a>(t: &'a ()) -> &'static ()
//~^ WARN unnecessary lifetime parameter `'a`
    where 'a: 'static { t }

fn static_id_indirect<'a,'b>(t: &'a ()) -> &'static ()
//~^ WARN unnecessary lifetime parameter `'a`
//~| WARN unnecessary lifetime parameter `'b`
    where 'a: 'b, 'b: 'static { t }

fn ref_id<'a>(t: &'a ()) -> &'a () where 'static: 'a { t }

static UNIT: () = ();

fn main()
{
    let mut val : &'static () = &UNIT;
    invariant_id(&mut val);
    static_id(val);
    static_id_indirect(val);
    ref_id(val);
}
