//@ build-pass (FIXME(62277): could be check-pass?)

fn main() {

    let _: &'static usize = &(loop {}, 1).1;
}
