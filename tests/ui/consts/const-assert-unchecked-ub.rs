const _: () = unsafe {
    let n = u32::MAX.count_ones();
    std::hint::assert_unchecked(n < 32); //~ ERROR `assume` called with `false`
};

fn main() {}
