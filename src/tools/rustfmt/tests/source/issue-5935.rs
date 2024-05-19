struct Regs<
    const BEGIN: u64,
    const END: u64,
    const DIM: usize,
    const N: usize = { (END - BEGIN) as usize / (8 * DIM) + 1 },
>
{
    _foo: u64,
}