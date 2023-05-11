// check-pass
const _: () = core::mem::forget(Box::<u32>::default);
const _: () = core::mem::forget(|| Box::<u32>::default());

fn main() {}
