#![feature(generic_const_items)]
#![expect(incomplete_features)]

const FOO<'a: 'static>: usize = 10;

fn bar<'a>() {
    match 10_usize {
        FOO::<'a> => todo!(),
        //~^ ERROR: lifetime may not live long enough
        _ => todo!(),
    }
}

fn main() {}
