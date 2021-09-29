// Compiler:
//
// Run-time:
//   status: 0

#![feature(asm, global_asm)]

global_asm!("
    .global add_asm
add_asm:
     mov rax, rdi
     add rax, rsi
     ret"
);

extern "C" {
    fn add_asm(a: i64, b: i64) -> i64;
}

fn main() {
    unsafe {
        asm!("nop");
    }

    let x: u64;
    unsafe {
        asm!("mov $5, {}",
            out(reg) x,
            options(att_syntax)
        );
    }
    assert_eq!(x, 5);

    let x: u64;
    let input: u64 = 42;
    unsafe {
        asm!("mov {input}, {output}",
             "add $1, {output}",
            input = in(reg) input,
            output = out(reg) x,
            options(att_syntax)
        );
    }
    assert_eq!(x, 43);

    let x: u64;
    unsafe {
        asm!("mov {}, 6",
            out(reg) x,
        );
    }
    assert_eq!(x, 6);

    let x: u64;
    let input: u64 = 42;
    unsafe {
        asm!("mov {output}, {input}",
             "add {output}, 1",
            input = in(reg) input,
            output = out(reg) x,
        );
    }
    assert_eq!(x, 43);

    // check inout(reg_class) x 
    let mut x: u64 = 42;
    unsafe {
        asm!("add {0}, {0}",
            inout(reg) x 
        );
    }
    assert_eq!(x, 84);

    // check inout("reg") x
    let mut x: u64 = 42;
    unsafe {
        asm!("add r11, r11",
            inout("r11") x 
        );
    }
    assert_eq!(x, 84);

    // check a mix of
    // in("reg")
    // inout(class) x => y
    // inout (class) x
    let x: u64 = 702;
    let y: u64 = 100;
    let res: u64;
    let mut rem: u64 = 0;
    unsafe {
        asm!("div r11",
            in("r11") y,
            inout("eax") x => res,
            inout("edx") rem,
        );
    }
    assert_eq!(res, 7);
    assert_eq!(rem, 2);

    // check const 
    let mut x: u64 = 42;
    unsafe {
        asm!("add {}, {}",
            inout(reg) x,
            const 1 
        );
    }
    assert_eq!(x, 43);

    // check const (ATT syntax)
    let mut x: u64 = 42;
    unsafe {
        asm!("add {}, {}",
            const 1,
            inout(reg) x,
            options(att_syntax)
        );
    }
    assert_eq!(x, 43);

    // check sym fn
    extern "C" fn foo() -> u64 { 42 }
    let x: u64;
    unsafe {
        asm!("call {}", sym foo, lateout("rax") x);
    }
    assert_eq!(x, 42);

    // check sym fn (ATT syntax)
    let x: u64;
    unsafe {
        asm!("call {}", sym foo, lateout("rax") x, options(att_syntax));
    }
    assert_eq!(x, 42);

    // check sym static
    static FOO: u64 = 42;
    let x: u64;
    unsafe {
        asm!("mov {1}, qword ptr [rip + {0}]", sym FOO, lateout(reg) x);
    }
    assert_eq!(x, 42);

    // check sym static (ATT syntax)
    let x: u64;
    unsafe {
        asm!("movq {0}(%rip), {1}", sym FOO, lateout(reg) x, options(att_syntax));
    }
    assert_eq!(x, 42);

    assert_eq!(unsafe { add_asm(40, 2) }, 42);
}
