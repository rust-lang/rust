// run-pass

#![feature(specialization)]

trait Iterator {
    fn next(&self);
}

trait WithAssoc {
    type Item;
}

impl<'a> WithAssoc for &'a () {
    type Item = &'a u32;
}

struct Cloned<I>(I);

impl<'a, I, T: 'a> Iterator for Cloned<I>
    where I: WithAssoc<Item=&'a T>, T: Clone
{
    fn next(&self) {}
}

impl<'a, I, T: 'a> Iterator for Cloned<I>
    where I: WithAssoc<Item=&'a T>, T: Copy
{

}

fn main() {
    Cloned(&()).next();
}
