#[no_mangle]
// EMIT_MIR slice_iter.built.after.mir
fn slice_iter_next<'a>(s_iter: &mut std::slice::Iter<'a, f32>) -> Option<&'a f32> {
    s_iter.next()
}
