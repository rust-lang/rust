#![cfg(not(feature = "no-asm"))]

use core::arch::global_asm;

global_asm!(include_str!("hexagon/func_macro.s"), options(raw));

global_asm!(include_str!("hexagon/dfaddsub.s"), options(raw));

global_asm!(include_str!("hexagon/dfdiv.s"), options(raw));

global_asm!(include_str!("hexagon/dffma.s"), options(raw));

global_asm!(include_str!("hexagon/dfminmax.s"), options(raw));

global_asm!(include_str!("hexagon/dfmul.s"), options(raw));

global_asm!(include_str!("hexagon/dfsqrt.s"), options(raw));

global_asm!(include_str!("hexagon/divdi3.s"), options(raw));

global_asm!(include_str!("hexagon/divsi3.s"), options(raw));

global_asm!(include_str!("hexagon/fastmath2_dlib_asm.s"), options(raw));

global_asm!(include_str!("hexagon/fastmath2_ldlib_asm.s"), options(raw));

global_asm!(
    include_str!("hexagon/memcpy_forward_vp4cp4n2.s"),
    options(raw)
);

global_asm!(
    include_str!("hexagon/memcpy_likely_aligned.s"),
    options(raw)
);

global_asm!(include_str!("hexagon/moddi3.s"), options(raw));

global_asm!(include_str!("hexagon/modsi3.s"), options(raw));

global_asm!(include_str!("hexagon/sfdiv_opt.s"), options(raw));

global_asm!(include_str!("hexagon/sfsqrt_opt.s"), options(raw));

global_asm!(include_str!("hexagon/udivdi3.s"), options(raw));

global_asm!(include_str!("hexagon/udivmoddi4.s"), options(raw));

global_asm!(include_str!("hexagon/udivmodsi4.s"), options(raw));

global_asm!(include_str!("hexagon/udivsi3.s"), options(raw));

global_asm!(include_str!("hexagon/umoddi3.s"), options(raw));

global_asm!(include_str!("hexagon/umodsi3.s"), options(raw));
