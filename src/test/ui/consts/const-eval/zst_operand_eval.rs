// build-pass (FIXME(62277): could be check-pass?)

static ASSERT: () = [()][!(std::mem::size_of::<u32>() == 4) as usize];

fn main() {}
