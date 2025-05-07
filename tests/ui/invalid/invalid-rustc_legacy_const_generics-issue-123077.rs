//@ only-x86_64

const fn foo<const U: i32>() -> i32 {
    U
}

fn main() {
    std::arch::x86_64::_mm_blend_ps(loop {}, loop {}, || ());
    //~^ ERROR invalid argument to a legacy const generic

    std::arch::x86_64::_mm_blend_ps(loop {}, loop {}, 5 + || ());
    //~^ ERROR invalid argument to a legacy const generic

    std::arch::x86_64::_mm_blend_ps(loop {}, loop {}, foo::<{ 1 + 2 }>());
    //~^ ERROR invalid argument to a legacy const generic

    std::arch::x86_64::_mm_blend_ps(loop {}, loop {}, foo::<3>());
    //~^ ERROR invalid argument to a legacy const generic

    std::arch::x86_64::_mm_blend_ps(loop {}, loop {}, &const {});
    //~^ ERROR invalid argument to a legacy const generic

    std::arch::x86_64::_mm_blend_ps(loop {}, loop {}, {
        struct F();
        //~^ ERROR invalid argument to a legacy const generic
        1
    });

    std::arch::x86_64::_mm_inserti_si64(loop {}, loop {}, || (), 1 + || ());
    //~^ ERROR invalid argument to a legacy const generic
}
