//@ only-linux
//@ needs-crate-type: cdylib

extern crate run_make_support;

use run_make_support::object::Object;
use run_make_support::{dynamic_lib_name, object, rfs, rustc};

const EXPORTED_SYMBOL: &[u8] = b"some$foo::bar$thing/path.rs:42";

fn main() {
    rustc().input("lib.rs").run();

    let contents = rfs::read(dynamic_lib_name("lib"));
    let object = object::File::parse(contents.as_slice()).unwrap();
    let matching_exports =
        object.exports().unwrap().iter().filter(|x| x.name() == EXPORTED_SYMBOL).count();
    assert_eq!(matching_exports, 1);
}
