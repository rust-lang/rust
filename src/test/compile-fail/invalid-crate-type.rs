// regression test for issue 11256
#[crate_type="foo"];    //~ ERROR invalid `crate_type` value

fn main() {
    return
}
