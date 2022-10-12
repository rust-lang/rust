// revisions: mirunsafeck thirunsafeck
// [thirunsafeck]compile-flags: -Z thir-unsafeck

union U {
    a: str,
    //~^ ERROR the size for values of type

    b: u8,
}

union W {
    a: u8,
    b: str,
    //~^ ERROR the size for values of type
}

fn main() {}
