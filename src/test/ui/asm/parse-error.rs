// only-x86_64

#![feature(asm, global_asm)]

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
        //~^ ERROR expected operand, options, or additional template string
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
        //~^ ERROR argument to `sym` must be a path expression
        asm!("", options(foo));
        //~^ ERROR expected one of
        asm!("", options(nomem foo));
        //~^ ERROR expected one of
        asm!("", options(nomem, foo));
        //~^ ERROR expected one of
        asm!("{}", options(), const foo);
        //~^ ERROR arguments are not allowed after options
        //~^^ ERROR attempt to use a non-constant value in a constant
        asm!("{a}", a = const foo, a = const bar);
        //~^ ERROR duplicate argument named `a`
        //~^^ ERROR argument never used
        //~^^^ ERROR attempt to use a non-constant value in a constant
        //~^^^^ ERROR attempt to use a non-constant value in a constant
        asm!("", a = in("eax") foo);
        //~^ ERROR explicit register arguments cannot have names
        asm!("{a}", in("eax") foo, a = const bar);
        //~^ ERROR named arguments cannot follow explicit register arguments
        //~^^ ERROR attempt to use a non-constant value in a constant
        asm!("{a}", in("eax") foo, a = const bar);
        //~^ ERROR named arguments cannot follow explicit register arguments
        //~^^ ERROR attempt to use a non-constant value in a constant
        asm!("{1}", in("eax") foo, const bar);
        //~^ ERROR positional arguments cannot follow named arguments or explicit register arguments
        //~^^ ERROR attempt to use a non-constant value in a constant
        asm!("", options(), "");
        //~^ ERROR expected one of
        asm!("{}", in(reg) foo, "{}", out(reg) foo);
        //~^ ERROR expected one of
        asm!(format!("{{{}}}", 0), in(reg) foo);
        //~^ ERROR asm template must be a string literal
        asm!("{1}", format!("{{{}}}", 0), in(reg) foo, out(reg) bar);
        //~^ ERROR asm template must be a string literal
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
//~^ ERROR arguments are not allowed after options
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
