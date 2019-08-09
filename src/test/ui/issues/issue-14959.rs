// build-pass (FIXME(62277): could be check-pass?)
// pretty-expanded FIXME #23616

#![feature(fn_traits, unboxed_closures)]

use std::ops::Fn;

trait Response { fn dummy(&self) { } }
trait Request { fn dummy(&self) { } }
trait Ingot<R, S> {
    fn enter(&mut self, _: &mut R, _: &mut S, a: &mut Alloy) -> Status;
}

#[allow(dead_code)]
struct HelloWorld;

struct SendFile;
struct Alloy;
enum Status {
    Continue
}

impl Alloy {
    fn find<T>(&self) -> Option<T> {
        None
    }
}

impl<'b> Fn<(&'b mut (dyn Response + 'b),)> for SendFile {
    extern "rust-call" fn call(&self, (_res,): (&'b mut (dyn Response + 'b),)) {}
}

impl<'b> FnMut<(&'b mut (dyn Response + 'b),)> for SendFile {
    extern "rust-call" fn call_mut(&mut self, (_res,): (&'b mut (dyn Response+'b),)) {
        self.call((_res,))
    }
}

impl<'b> FnOnce<(&'b mut (dyn Response + 'b),)> for SendFile {
    type Output = ();

    extern "rust-call" fn call_once(self, (_res,): (&'b mut (dyn Response+'b),)) {
        self.call((_res,))
    }
}

impl<Rq: Request, Rs: Response> Ingot<Rq, Rs> for HelloWorld {
    fn enter(&mut self, _req: &mut Rq, res: &mut Rs, alloy: &mut Alloy) -> Status {
        let send_file = alloy.find::<SendFile>().unwrap();
        send_file(res);
        Status::Continue
    }
}

fn main() {}
