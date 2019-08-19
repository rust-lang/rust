// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]

trait Resources {
    type Buffer: Copy;
}

#[derive(Copy, Clone)]
struct ConstantBufferSet<R: Resources>(
    pub R::Buffer
);

#[derive(Copy, Clone)]
enum It {}
impl Resources for It {
    type Buffer = u8;
}

#[derive(Copy, Clone)]
enum Command {
    BindConstantBuffers(ConstantBufferSet<It>)
}

fn main() {}
