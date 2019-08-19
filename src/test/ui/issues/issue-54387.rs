// build-pass (FIXME(62277): could be check-pass?)

pub struct GstRc {
    _obj: *const (),
    _borrowed: bool,
}

const FOO: Option<GstRc> = None;

fn main() {
    let _meh = FOO;
}
