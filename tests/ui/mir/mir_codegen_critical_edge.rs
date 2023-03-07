// run-pass
#![allow(dead_code)]
// This code produces a CFG with critical edges that, if we don't
// handle properly, will cause invalid codegen.

#![feature(rustc_attrs)]

enum State {
    Both,
    Front,
    Back
}

pub struct Foo<A: Iterator, B: Iterator> {
    state: State,
    a: A,
    b: B
}

impl<A, B> Foo<A, B>
where A: Iterator, B: Iterator<Item=A::Item>
{
    // This is the function we care about
    fn next(&mut self) -> Option<A::Item> {
        match self.state {
            State::Both => match self.a.next() {
                elt @ Some(..) => elt,
                None => {
                    self.state = State::Back;
                    self.b.next()
                }
            },
            State::Front => self.a.next(),
            State::Back => self.b.next(),
        }
    }
}

// Make sure we actually codegen a version of the function
pub fn do_stuff(mut f: Foo<Box<dyn Iterator<Item=u32>>, Box<dyn Iterator<Item=u32>>>) {
    let _x = f.next();
}

fn main() {}
