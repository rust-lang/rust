// rustfmt-normalize_comments: true
fn types() {
    let x: [Vec<_>] = [];
    let y: *mut [SomeType; konst_funk()] = expr();
    let z: (/* #digits */ usize, /* exp */ i16) = funk();
    let z: (usize /* #digits */, i16 /* exp */) = funk();
}

struct F {
    f: extern "C" fn(x: u8, ... /* comment */),
    g: extern "C" fn(x: u8, /* comment */ ...),
    h: extern "C" fn(x: u8, ...),
    i: extern "C" fn(
        x: u8,
        // comment 4
        y: String, // comment 3
        z: Foo,
        // comment
        ... // comment 2
    ),
}

fn issue_1006(def_id_to_string: for<'a, 'b> unsafe fn(TyCtxt<'b, 'tcx, 'tcx>, DefId) -> String) {}

fn impl_trait_fn_1() -> impl Fn(i32) -> Option<u8> {}

fn impl_trait_fn_2<E>() -> impl Future<Item = &'a i64, Error = E> {}

fn issue_1234() {
    do_parse!(name: take_while1!(is_token) >> (Header))
}
