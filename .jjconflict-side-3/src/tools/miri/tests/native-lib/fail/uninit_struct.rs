#[repr(C)]
#[derive(Copy, Clone)]
struct ComplexStruct {
    part_1: Part1,
    part_2: Part2,
    part_3: u32,
}
#[repr(C)]
#[derive(Copy, Clone)]
struct Part1 {
    high: u16,
    low: u16,
}
#[repr(C)]
#[derive(Copy, Clone)]
struct Part2 {
    bits: u32,
}

extern "C" {
    fn pass_struct_complex(s: ComplexStruct, high: u16, low: u16, bits: u32) -> i32;
}

fn main() {
    let arg = std::mem::MaybeUninit::<ComplexStruct>::uninit();
    unsafe { pass_struct_complex(*arg.as_ptr(), 0, 0, 0) }; //~ ERROR: Undefined Behavior: constructing invalid value
}
