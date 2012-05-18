
// check that we do not report a type like this as uninstantiable,
// even though it would be if the nxt field had type @foo:
enum foo = {x: uint, nxt: *foo};

fn main() {
    let x = foo({x: 0u, nxt: ptr::null()});
}

