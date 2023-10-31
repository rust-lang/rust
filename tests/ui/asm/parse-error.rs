// needs-asm-support

#![feature(asm_const)]

use std::arch::{asm, global_asm};

fn main() {
    let mut foo = 0;
    let mut bar = 0;
    unsafe {
        asm!();
        //~^ ERROR requires at least a template string argument
        asm!(foo);
        //~^ ERROR asm template must be a string literal
        asm!("{}" foo);
        //~^ ERROR expected token: `,`
        asm!("{}", foo);
        //~^ ERROR expected operand, clobber_abi, options, or additional template string
        asm!("{}", in foo);
        //~^ ERROR expected `(`, found `foo`
        asm!("{}", in(reg foo));
        //~^ ERROR expected `)`, found `foo`
        asm!("{}", in(reg));
        //~^ ERROR expected expression, found end of macro arguments
        asm!("{}", inout(=) foo => bar);
        //~^ ERROR expected register class or explicit register
        asm!("{}", inout(reg) foo =>);
        //~^ ERROR expected expression, found end of macro arguments
        asm!("{}", in(reg) foo => bar);
        //~^ ERROR expected one of `!`, `,`, `.`, `::`, `?`, `{`, or an operator, found `=>`
        asm!("{}", sym foo + bar);
        //~^ ERROR expected a path for argument to `sym`
        asm!("", options(foo));
        //~^ ERROR expected one of
        asm!("", options(nomem foo));
        //~^ ERROR expected one of
        asm!("", options(nomem, foo));
        //~^ ERROR expected one of
        asm!("{}", options(), const foo);
        //~^ ERROR attempt to use a non-constant value in a constant

        // test that asm!'s clobber_abi doesn't accept non-string literals
        // see also https://github.com/rust-lang/rust/issues/112635
        asm!("", clobber_abi());
        //~^ ERROR at least one abi must be provided
        asm!("", clobber_abi(foo));
        //~^ ERROR expected string literal
        asm!("", clobber_abi("C" foo));
        //~^ ERROR expected one of `)` or `,`, found `foo`
        asm!("", clobber_abi("C", foo));
        //~^ ERROR expected string literal
        asm!("", clobber_abi(1));
        //~^ ERROR expected string literal
        asm!("", clobber_abi(()));
        //~^ ERROR expected string literal
        asm!("", clobber_abi(uwu));
        //~^ ERROR expected string literal
        asm!("", clobber_abi({}));
        //~^ ERROR expected string literal
        asm!("", clobber_abi(loop {}));
        //~^ ERROR expected string literal
        asm!("", clobber_abi(if));
        //~^ ERROR expected string literal
        asm!("", clobber_abi(do));
        //~^ ERROR expected string literal
        asm!("", clobber_abi(<));
        //~^ ERROR expected string literal
        asm!("", clobber_abi(.));
        //~^ ERROR expected string literal

        asm!("{}", clobber_abi("C"), const foo);
        //~^ ERROR attempt to use a non-constant value in a constant
        asm!("", options(), clobber_abi("C"));
        asm!("{}", options(), clobber_abi("C"), const foo);
        //~^ ERROR attempt to use a non-constant value in a constant
        asm!("{a}", a = const foo, a = const bar);
        //~^ ERROR duplicate argument named `a`
        //~^^ ERROR argument never used
        //~^^^ ERROR attempt to use a non-constant value in a constant
        //~^^^^ ERROR attempt to use a non-constant value in a constant

        asm!("", options(), "");
        //~^ ERROR expected one of
        asm!("{}", in(reg) foo, "{}", out(reg) foo);
        //~^ ERROR expected one of
        asm!(format!("{{{}}}", 0), in(reg) foo);
        //~^ ERROR asm template must be a string literal
        asm!("{1}", format!("{{{}}}", 0), in(reg) foo, out(reg) bar);
        //~^ ERROR asm template must be a string literal
        asm!("{}", in(reg) _);
        //~^ ERROR _ cannot be used for input operands
        asm!("{}", inout(reg) _);
        //~^ ERROR _ cannot be used for input operands
        asm!("{}", inlateout(reg) _);
        //~^ ERROR _ cannot be used for input operands
    }
}

const FOO: i32 = 1;
const BAR: i32 = 2;
global_asm!();
//~^ ERROR requires at least a template string argument
global_asm!(FOO);
//~^ ERROR asm template must be a string literal
global_asm!("{}" FOO);
//~^ ERROR expected token: `,`
global_asm!("{}", FOO);
//~^ ERROR expected operand, options, or additional template string
global_asm!("{}", const);
//~^ ERROR expected expression, found end of macro arguments
global_asm!("{}", const(reg) FOO);
//~^ ERROR expected one of
global_asm!("", options(FOO));
//~^ ERROR expected one of
global_asm!("", options(nomem FOO));
//~^ ERROR expected one of
global_asm!("", options(nomem, FOO));
//~^ ERROR expected one of
global_asm!("{}", options(), const FOO);
global_asm!("", clobber_abi(FOO));
//~^ ERROR expected string literal
global_asm!("", clobber_abi("C" FOO));
//~^ ERROR expected one of `)` or `,`, found `FOO`
global_asm!("", clobber_abi("C", FOO));
//~^ ERROR expected string literal
global_asm!("{}", clobber_abi("C"), const FOO);
//~^ ERROR `clobber_abi` cannot be used with `global_asm!`
global_asm!("", options(), clobber_abi("C"));
//~^ ERROR `clobber_abi` cannot be used with `global_asm!`
global_asm!("{}", options(), clobber_abi("C"), const FOO);
//~^ ERROR `clobber_abi` cannot be used with `global_asm!`
global_asm!("", clobber_abi("C"), clobber_abi("C"));
//~^ ERROR `clobber_abi` cannot be used with `global_asm!`
global_asm!("{a}", a = const FOO, a = const BAR);
//~^ ERROR duplicate argument named `a`
//~^^ ERROR argument never used
global_asm!("", options(), "");
//~^ ERROR expected one of
global_asm!("{}", const FOO, "{}", const FOO);
//~^ ERROR expected one of
global_asm!(format!("{{{}}}", 0), const FOO);
//~^ ERROR asm template must be a string literal
global_asm!("{1}", format!("{{{}}}", 0), const FOO, const BAR);
//~^ ERROR asm template must be a string literal
