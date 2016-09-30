#![feature(lang_items)]
#![no_std]

extern {
    fn __ashldi3();
    fn __ashrdi3();
    fn __divdi3();
    fn __divmoddi4();
    fn __divmodsi4();
    fn __divsi3();
    fn __lshrdi3();
    fn __moddi3();
    fn __modsi3();
    fn __muldi3();
    fn __mulodi4();
    fn __mulosi4();
    fn __udivdi3();
    fn __udivmoddi4();
    fn __udivmodsi4();
    fn __udivsi3();
    fn __umoddi3();
    fn __umodsi3();
    fn __addsf3();
    fn __adddf3();
    fn __powisf2();
    fn __powidf2();
}

macro_rules! declare {
    ($func:ident, $sym:ident) => {
        #[no_mangle]
        pub extern fn $func() -> usize {
            $sym as usize
        }
    }
}

declare!(___ashldi3, __ashldi3);
declare!(___ashrdi3, __ashrdi3);
declare!(___divdi3, __divdi3);
declare!(___divmoddi4, __divmoddi4);
declare!(___divmodsi4, __divmodsi4);
declare!(___divsi3, __divsi3);
declare!(___lshrdi3, __lshrdi3);
declare!(___moddi3, __moddi3);
declare!(___modsi3, __modsi3);
declare!(___muldi3, __muldi3);
declare!(___mulodi4, __mulodi4);
declare!(___mulosi4, __mulosi4);
declare!(___udivdi3, __udivdi3);
declare!(___udivmoddi4, __udivmoddi4);
declare!(___udivmodsi4, __udivmodsi4);
declare!(___udivsi3, __udivsi3);
declare!(___umoddi3, __umoddi3);
declare!(___umodsi3, __umodsi3);
declare!(___addsf3, __addsf3);
declare!(___adddf3, __adddf3);
declare!(___powisf2, __powisf2);
declare!(___powidf2, __powidf2);

#[lang = "eh_personality"]
fn eh_personality() {}
#[lang = "panic_fmt"]
fn panic_fmt() {}
